#include "erl_common/eigen.hpp"

int
mkl_serv_intel_cpu_true() {  // cheat MKL to think it is running on Intel CPU
    return 1;
}

namespace erl::common {

    Eigen::IOFormat
    GetEigenTextFormat(EigenTextFormat format) {
        static const Eigen::IOFormat kFormats[] = {
            Eigen::IOFormat(),         // kDefaultFmt
            Eigen::IOFormat(           // kCommaInitFmt
                Eigen::FullPrecision,  // precision
                0,                     // flags
                ", ",                  // coeffSeparator
                ",\n",                 // rowSeparator
                "",                    // rowPrefix
                "",                    // rowSuffix
                "<<\n ",               // matPrefix
                ";"                    // matSuffix
                ),
            Eigen::IOFormat(           // kCleanFmt
                Eigen::FullPrecision,  // precision
                0,                     // flags
                ", ",                  // coeffSeparator
                "\n",                  // rowSeparator
                "[",                   // rowPrefix
                "]"                    // rowSuffix
                ),
            Eigen::IOFormat(           // kOctaveFmt
                Eigen::FullPrecision,  // precision
                0,                     // flags
                ", ",                  // coeffSeparator
                ";\n",                 // rowSeparator
                "",                    // rowPrefix
                "",                    // rowSuffix
                "[",                   // matPrefix
                "]"                    // matSuffix
                ),
            Eigen::IOFormat(           // kNumpyFmt
                Eigen::FullPrecision,  // precision
                0,                     // flags
                ", ",                  // coeffSeparator
                ",\n",                 // rowSeparator
                "[",                   // rowPrefix
                "]",                   // rowSuffix
                "[",                   // matPrefix
                "]"                    // matSuffix
                ),
            Eigen::IOFormat(           // kCsvFmt
                Eigen::FullPrecision,  // precision
                0,                     // flags
                ", ",                  // coeffSeparator
                "\n"                   // rowSeparator
                )};
        return kFormats[static_cast<int>(format)];
    }
}  // namespace erl::common
