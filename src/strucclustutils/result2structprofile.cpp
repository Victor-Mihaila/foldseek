#include <cassert>
#include <cstdlib>
#include <string>

#include "CommandCaller.h"
#include "DBReader.h"
#include "DBWriter.h"
#include "Debug.h"
#include "FileUtil.h"
#include "LocalParameters.h"
#include "Matcher.h"
#include "PrefilteringIndexReader.h"
#include "Sequence.h"
#include "StructureProfileUtils.h"
#include "StructureUtil.h"
#include "Util.h"
#include "result2structprofile.sh.h"

#include <algorithm>
#include <cctype>
#include <vector>

#ifdef OPENMP
#include <omp.h>
#endif

int result2structprofile(int argc, const char **argv, const Command &command) {
    LocalParameters &par = LocalParameters::getLocalInstance();
    par.parseParameters(argc, argv, command, true, Parameters::PARSE_VARIADIC, 0);

    CommandCaller cmd;
    const std::string db2NoIndexName = PrefilteringIndexReader::dbPathWithoutIndex(par.db2);

    const char *disableSs12ProfileEnv = std::getenv("FOLDSEEK_DISABLE_SS12_PROFILE");
    const bool disableSs12Profile = (disableSs12ProfileEnv != NULL
                                     && disableSs12ProfileEnv[0] != '\0'
                                     && disableSs12ProfileEnv[0] != '0');
    const ResolvedStructureProfileDbs resolved = resolveStructureProfileDbs(par, db2NoIndexName, disableSs12Profile);

    if (FileUtil::fileExists((par.db1 + ".dbtype").c_str())) {
        cmd.addVariable("PROFILE_PAR", createAaProfileParameters(par).c_str());
    }

    if (FileUtil::fileExists((resolved.ssInputDb + ".dbtype").c_str())) {
        const std::string ssTargetDb = resolved.ssTargetDb + (db2NoIndexName != par.db2 ? ".idx" : "");
        cmd.addVariable("INPUT_SS_DB", resolved.ssInputDb.c_str());
        cmd.addVariable("TARGET_SS_DB", ssTargetDb.c_str());
        cmd.addVariable("PROFILE_SS_PAR", create3DiProfileParameters(par).c_str());
    }

    if (resolved.ss12InputDb.empty() == false && resolved.ss12TargetDb.empty() == false) {
        cmd.addVariable("INPUT_SS12_DB", resolved.ss12InputDb.c_str());
        cmd.addVariable("TARGET_SS12_DB", resolved.ss12TargetDb.c_str());
        cmd.addVariable("PROFILE_SS12_PAR", create12StateProfileParameters(par).c_str());
        cmd.addVariable("CREATE_SS12_PROFILE", "1");
        if (par.profileOutputMode == 1) {
            cmd.addVariable("PROFILE_SS12_OUTPUT_COLUMNS", "12");
            cmd.addVariable("PROFILE_SS12_OUTPUT_LABELS", "A C D E F G H I K L M N");
        }
    }

    if (resolved.removeSplitTmp) {
        cmd.addVariable("REMOVE_SPLIT_TMP", "1");
    }

    if (par.profileLowconfMembers == 1) {
        cmd.addVariable("LOWCONF_TARGET_DB", db2NoIndexName.c_str());
        cmd.addVariable("LOWCONF_PAR", par.createParameterString(par.lowconfmembers).c_str());
    }

    if (par.profileLowconfQuery > 0) {
        if (par.profileOutputMode == 1) {
            Debug(Debug::ERROR) << "--profile-lowconf-query works on binary profiles only (--profile-output-mode 0)\n";
            EXIT(EXIT_FAILURE);
        }
        cmd.addVariable("LOWCONFQ", "1");
        std::vector<MMseqsParameter *> lcq;
        lcq.push_back(&par.PARAM_THREADS);
        lcq.push_back(&par.PARAM_V);
        cmd.addVariable("LOWCONFQ_PAR", par.createParameterString(lcq).c_str());
    }

    cmd.addVariable("VERBOSITY", par.createParameterString(par.onlyverbosity).c_str());

    const std::string program = par.db4 + ".sh";
    FileUtil::writeFile(program, result2structprofile_sh, result2structprofile_sh_len);
    cmd.execProgram(FileUtil::getRealPathFromSymLink(program).c_str(), par.filenames);

    assert(false);
    return EXIT_FAILURE;
}


int lowconfmembers(int argc, const char **argv, const Command &command) {
    LocalParameters &par = LocalParameters::getLocalInstance();
    par.parseParameters(argc, argv, command, true, 0, 0);

    DBReader<unsigned int> tDbr(par.db1.c_str(), par.db1Index.c_str(), par.threads,
                                DBReader<unsigned int>::USE_INDEX | DBReader<unsigned int>::USE_DATA);
    tDbr.open(DBReader<unsigned int>::NOSORT);
    if (Parameters::isEqualDbtype(tDbr.getDbtype(), Parameters::DBTYPE_AMINO_ACIDS) == false) {
        Debug(Debug::ERROR) << "lowconfmembers: " << par.db1 << " is not an amino-acid DB; the pLDDT marks are its lowercase letters\n";
        EXIT(EXIT_FAILURE);
    }
    DBReader<unsigned int> alnDbr(par.db2.c_str(), par.db2Index.c_str(), par.threads,
                                  DBReader<unsigned int>::USE_INDEX | DBReader<unsigned int>::USE_DATA);
    alnDbr.open(DBReader<unsigned int>::LINEAR_ACCCESS);
    DBWriter writer(par.db3.c_str(), par.db3Index.c_str(), par.threads, par.compressed, alnDbr.getDbtype());
    writer.open();

    const float maxLowFrac = par.lowconfMemberMax;
    size_t nMembers = 0, nDropped = 0, nMatched = 0, nGapped = 0;
    Debug::Progress progress(alnDbr.getSize());

#pragma omp parallel num_threads(par.threads) reduction(+:nMembers, nDropped, nMatched, nGapped)
    {
        unsigned int thread_idx = 0;
#ifdef OPENMP
        thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
        std::string out;
        std::string bt;
        std::vector<std::string> fields;

#pragma omp for schedule(dynamic, 10)
        for (size_t id = 0; id < alnDbr.getSize(); id++) {
            progress.updateProgress();
            const unsigned int queryKey = alnDbr.getDbKey(id);
            char *data = alnDbr.getData(id, thread_idx);
            out.clear();
            while (*data != '\0') {
                const char *lineStart = data;
                Matcher::result_t res = Matcher::parseAlignmentRecord(data, false);
                data = Util::skipLine(data);
                const char *lineEnd = data;
                while (lineEnd > lineStart && (lineEnd[-1] == '\n' || lineEnd[-1] == '\0')) {
                    lineEnd--;
                }
                if (res.backtrace.empty()) {
                    Debug(Debug::ERROR) << "lowconfmembers: alignment of query " << queryKey << " to " << res.dbKey
                                        << " has no backtrace; the alignment DB must be computed with -a\n";
                    EXIT(EXIT_FAILURE);
                }
                nMembers++;
                size_t low = 0;
                if (res.qStartPos <= res.qEndPos && res.dbStartPos <= res.dbEndPos) {
                    const size_t tId = tDbr.getId(res.dbKey);
                    if (tId == UINT_MAX) {
                        Debug(Debug::ERROR) << "lowconfmembers: target " << res.dbKey << " is not in " << par.db1 << "\n";
                        EXIT(EXIT_FAILURE);
                    }
                    const char *tSeq = tDbr.getData(tId, thread_idx);
                    int tPos = res.dbStartPos;
                    size_t matched = 0;
                    bt.clear();
                    for (char c : res.backtrace) {
                        if (c == 'M') {
                            matched++;
                            if (std::islower(static_cast<unsigned char>(tSeq[tPos]))) {
                                low++;
                                bt += "DI";
                            } else {
                                bt += 'M';
                            }
                            tPos++;
                        } else {
                            bt += c;
                            if (c == 'D') {
                                tPos++;
                            }
                        }
                    }
                    if (matched == 0 || low == matched || low > maxLowFrac * matched) {
                        nDropped++;
                        continue;
                    }
                    nMatched += matched;
                    nGapped += low;
                }
                if (low == 0) {
                    out.append(lineStart, lineEnd - lineStart);
                    out.push_back('\n');
                    continue;
                }
                size_t b = 0, e = bt.size();
                while (b < e && bt[b] != 'M') {
                    if (bt[b] == 'D') { res.dbStartPos++; } else { res.qStartPos++; }
                    b++;
                }
                while (e > b && bt[e - 1] != 'M') {
                    if (bt[e - 1] == 'D') { res.dbEndPos--; } else { res.qEndPos--; }
                    e--;
                }
                fields.clear();
                const char *f = lineStart;
                for (const char *c = lineStart; c <= lineEnd; c++) {
                    if (c == lineEnd || *c == '\t') {
                        fields.emplace_back(f, c - f);
                        f = c + 1;
                    }
                }
                fields[4] = SSTR(res.qStartPos);
                fields[5] = SSTR(res.qEndPos);
                fields[7] = SSTR(res.dbStartPos);
                fields[8] = SSTR(res.dbEndPos);
                fields.back() = Matcher::compressAlignment(bt.substr(b, e - b));
                for (size_t i = 0; i < fields.size(); i++) {
                    out.append(fields[i]);
                    out.push_back(i + 1 < fields.size() ? '\t' : '\n');
                }
            }
            writer.writeData(out.c_str(), out.size(), queryKey, thread_idx);
        }
    }
    writer.close(true);
    alnDbr.close();
    tDbr.close();

    const double pct = nMatched > 0 ? 100.0 * nGapped / nMatched : 0.0;
    Debug(Debug::INFO) << "lowconfmembers: " << nMembers << " member alignments, " << nDropped
                       << " dropped (> " << maxLowFrac << " of matched residues low-confidence); "
                       << nGapped << " of " << nMatched << " matched residues of the kept members gapped ("
                       << pct << "%)\n";
    if (nMembers > 0 && nDropped == 0 && nGapped == 0) {
        Debug(Debug::WARNING) << "lowconfmembers: no member residue is lowercase. The target DB carries no pLDDT marks "
                                 "(build it with --mask-bfactor-threshold), so --profile-lowconf-members has no effect\n";
    }
    return EXIT_SUCCESS;
}

int lowconfquery(int argc, const char **argv, const Command &command) {
    LocalParameters &par = LocalParameters::getLocalInstance();
    par.parseParameters(argc, argv, command, true, 0, 0);

    DBReader<unsigned int> qDbr(par.db1.c_str(), par.db1Index.c_str(), par.threads,
                                DBReader<unsigned int>::USE_INDEX | DBReader<unsigned int>::USE_DATA);
    qDbr.open(DBReader<unsigned int>::NOSORT);
    if (Parameters::isEqualDbtype(qDbr.getDbtype(), Parameters::DBTYPE_AMINO_ACIDS) == false) {
        Debug(Debug::ERROR) << "lowconfquery: " << par.db1 << " is not an amino-acid db; the pLDDT marks are its lowercase letters\n";
        EXIT(EXIT_FAILURE);
    }
    DBReader<unsigned int> pDbr(par.db2.c_str(), par.db2Index.c_str(), par.threads,
                                DBReader<unsigned int>::USE_INDEX | DBReader<unsigned int>::USE_DATA);
    pDbr.open(DBReader<unsigned int>::LINEAR_ACCCESS);
    const bool isProfile = Parameters::isEqualDbtype(pDbr.getDbtype(), Parameters::DBTYPE_HMM_PROFILE);
    if (isProfile == false && StructureUtil::is3Di12StDb(pDbr.getDbtype()) == false) {
        Debug(Debug::ERROR) << "lowconfquery: " << par.db2 << " is neither a profile db nor a packed 3Di+12-state db\n";
        EXIT(EXIT_FAILURE);
    }
    const char *what = isProfile ? "profile positions" : "3Di+12-state residues";
    DBWriter writer(par.db3.c_str(), par.db3Index.c_str(), par.threads, par.compressed, pDbr.getDbtype());
    writer.open();

    size_t nPos = 0, nZeroed = 0;
    Debug::Progress progress(pDbr.getSize());

#pragma omp parallel num_threads(par.threads) reduction(+:nPos, nZeroed)
    {
        unsigned int thread_idx = 0;
#ifdef OPENMP
        thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
        std::vector<unsigned char> rec;
#pragma omp for schedule(dynamic, 10)
        for (size_t id = 0; id < pDbr.getSize(); id++) {
            progress.updateProgress();
            const unsigned int key = pDbr.getDbKey(id);
            const unsigned char *data = reinterpret_cast<const unsigned char *>(pDbr.getData(id, thread_idx));
            const size_t len = pDbr.getEntryLen(id) - 1;
            const size_t L = isProfile ? len / Sequence::PROFILE_READIN_SIZE : len - 1;
            rec.assign(data, data + len);
            const size_t qId = qDbr.getId(key);
            if (qId == UINT_MAX) {
                Debug(Debug::ERROR) << "lowconfquery: query " << key << " is not in " << par.db1 << "\n";
                EXIT(EXIT_FAILURE);
            }
            const char *qSeq = qDbr.getData(qId, thread_idx);
            const size_t qLen = qDbr.getSeqLen(qId);
            if (qLen != L) {
                Debug(Debug::ERROR) << "lowconfquery: query " << key << " has " << qLen << " residues but " << par.db2
                                    << " " << L << "\n";
                EXIT(EXIT_FAILURE);
            }
            nPos += L;
            for (size_t i = 0; i < L; ++i) {
                if (islower(static_cast<unsigned char>(qSeq[i]))) {
                    if (isProfile) {
                        std::fill_n(rec.data() + i * Sequence::PROFILE_READIN_SIZE, Sequence::PROFILE_AA_SIZE, 0);
                    } else {
                        rec[i] = Alphabet12St::PACKED_X;
                    }
                    nZeroed++;
                }
            }
            writer.writeData(reinterpret_cast<const char *>(rec.data()), rec.size(), key, thread_idx);
        }
    }
    writer.close(true);
    pDbr.close();
    qDbr.close();

    Debug(Debug::INFO) << "lowconfquery: " << nZeroed << " of " << nPos << " " << what
                       << " at low-confidence query residues " << (isProfile ? "zeroed" : "set to X") << " (" << par.db2 << ")\n";
    if (nPos > 0 && nZeroed == 0) {
        Debug(Debug::WARNING) << "lowconfquery: no query residue is lowercase. The query db carries no pLDDT marks "
                                 "(build it with --mask-bfactor-threshold), so nothing was masked\n";
    }
    return EXIT_SUCCESS;
}
