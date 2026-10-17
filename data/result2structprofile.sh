#!/bin/sh -e

INPUT="$1"
TARGET="$2"
RESULT="$3"
OUT="$4"
INPUT_SS_DB="${INPUT_SS_DB:-${INPUT}_ss}"
TARGET_SS_DB="${TARGET_SS_DB:-${TARGET}_ss${INDEXEXT}}"

fail() {
    echo "$1" >&2
    exit 1
}

if [ -n "${LOWCONF_PAR}" ]; then
    # shellcheck disable=SC2086
    "$MMSEQS" lowconfmembers "${LOWCONF_TARGET_DB}" "${RESULT}" "${OUT}_lcaln" ${LOWCONF_PAR} \
        || die "lowconfmembers died"
    RESULT="${OUT}_lcaln"
    REMOVE_LCALN=1
fi

if [ -e "${INPUT}.dbtype" ]; then
    # shellcheck disable=SC2086
    $RUNNER "$MMSEQS" base:result2profile "${INPUT}" "${TARGET}${INDEXEXT}" "${RESULT}" "${OUT}" ${PROFILE_PAR} \
        || fail "result2profile for AA died"
fi

if [ -e "${INPUT_SS_DB}.dbtype" ]; then
    # shellcheck disable=SC2086
    $RUNNER "$MMSEQS" base:result2profile "${INPUT_SS_DB}" "${TARGET_SS_DB}" "${RESULT}" "${OUT}_ss" ${PROFILE_SS_PAR} \
        || fail "result2profile for 3Di died"
fi

if [ -n "${CREATE_SS12_PROFILE}" ]; then
    SS12_OUT="${OUT}_ss12"
    SS12_RAW_OUT="${SS12_OUT}"
    if [ -n "${PROFILE_SS12_OUTPUT_COLUMNS}" ]; then
        SS12_RAW_OUT="${OUT}_ss12.raw"
    fi

    # shellcheck disable=SC2086
    $RUNNER "$MMSEQS" base:result2profile "${INPUT_SS12_DB}" "${TARGET_SS12_DB}" "${RESULT}" "${SS12_RAW_OUT}" ${PROFILE_SS12_PAR} \
        || fail "result2profile for 12-state died"

    if [ -n "${PROFILE_SS12_OUTPUT_COLUMNS}" ]; then
        awk -v keep="${PROFILE_SS12_OUTPUT_COLUMNS}" \
            -v labels="${PROFILE_SS12_OUTPUT_LABELS:-A C D E F G H I K L M N}" '
            /^Query profile of sequence / {
                print
                expect_header = 1
                next
            }
            expect_header {
                n = split(labels, label_parts, /[[:space:]]+/)
                printf "     "
                for (i = 1; i <= keep && i <= n; i++) {
                    if (i == keep || i == n) {
                        printf "%s\n", label_parts[i]
                    } else {
                        printf "%-7s", label_parts[i]
                    }
                }
                expect_header = 0
                next
            }
            NF == 0 {
                print
                next
            }
            {
                sum = 0.0
                for (i = 1; i <= keep && i <= NF; i++) {
                    sum += $i
                }
                for (i = 1; i <= keep && i <= NF; i++) {
                    value = (sum > 0.0) ? $i / sum : 0.0
                    printf "%.4f%s", value, (i == keep || i == NF) ? "\n" : " "
                }
            }
        ' "${SS12_RAW_OUT}" > "${SS12_OUT}" || fail "12-state profile formatting died"
        rm -f -- "${SS12_RAW_OUT}"
    fi
fi

if [ -n "${LOWCONFQ}" ]; then
    lcq() {
        # shellcheck disable=SC2086,SC2153
        "$MMSEQS" lowconfquery "${INPUT}" "$1" "$1_lcq" ${LOWCONFQ_PAR} || die "lowconfquery died"
        mv -f -- "$1_lcq" "$1" || die "lowconfquery: could not replace $1"
        mv -f -- "$1_lcq.index" "$1.index" || die "lowconfquery: could not replace $1.index"
        rm -f -- "$1_lcq.dbtype"
    }
    if [ -e "${OUT}_ss.dbtype" ]; then lcq "${OUT}_ss"; fi
    if [ -e "${OUT}_ss12.dbtype" ]; then lcq "${OUT}_ss12"; fi
fi

if [ -e "${INPUT}_ca.dbtype" ]; then
    # shellcheck disable=SC2086
    "$MMSEQS" lndb "${INPUT}_ca" "${OUT}_ca" ${VERBOSITY} \
        || fail "Create lndb died"
fi

if [ -e "${INPUT}_h.dbtype" ]; then
    # shellcheck disable=SC2086
    "$MMSEQS" lndb "${INPUT}_h" "${OUT}_h" ${VERBOSITY} \
        || fail "Create lndb died"
fi

if [ -e "${INPUT}.sh" ]; then
    rm -f -- "${OUT}.sh"
fi

if [ -n "${REMOVE_LCALN}" ]; then
    # shellcheck disable=SC2086
    "$MMSEQS" rmdb "${OUT}_lcaln" ${VERBOSITY} || true
fi

if [ -n "${REMOVE_SPLIT_QUERY}" ]; then
    # shellcheck disable=SC2086
    "$MMSEQS" rmdb "${INPUT_SS_DB}" ${VERBOSITY} || true
    if [ -n "${INPUT_SS12_DB}" ]; then
        # shellcheck disable=SC2086
        "$MMSEQS" rmdb "${INPUT_SS12_DB}" ${VERBOSITY} || true
    fi
fi

if [ -n "${REMOVE_SPLIT_TARGET}" ]; then
    # shellcheck disable=SC2086
    "$MMSEQS" rmdb "${TARGET_SS_DB}" ${VERBOSITY} || true
    if [ -n "${TARGET_SS12_DB}" ]; then
        # shellcheck disable=SC2086
        "$MMSEQS" rmdb "${TARGET_SS12_DB}" ${VERBOSITY} || true
    fi
fi
