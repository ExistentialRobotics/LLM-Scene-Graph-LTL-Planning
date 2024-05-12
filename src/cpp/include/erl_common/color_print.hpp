#pragma once

#include <sstream>
#include <vector>

namespace erl::common {

    // http://www.isthe.com/chongo/tech/comp/ansi_escapes.html

    enum TermCode {
        kTermFgBlack = 30,
        kTermFgRed = 31,
        kTermFgGreen = 32,
        kTermFgYellow = 33,
        kTermFgBlue = 34,
        kTermFgMagenta = 35,
        kTermFgCyan = 36,
        kTermFgWhite = 37,
        kTermFgDefault = 39,
        kTermBgBlack = 40,
        kTermBgRed = 41,
        kTermBgGreen = 42,
        kTermBgYellow = 43,
        kTermBgBlue = 44,
        kTermBgMagenta = 45,
        kTermBgCyan = 46,
        kTermBgWhite = 47,
        kTermBgDefault = 49,
        kTermRest = 0,
        kTermBold = 1,
        kTermUnderline = 4,
        kTermBlink = 5,
        kTermInverse = 7,
        kTermInvisible = 8,
        kTermBoldOff = 21,
        kTermNormal = 22,
        kTermStandoutOff = 23,
        kTermUnderlineOff = 24,
        kTermBlinkOff = 25,
        kTermInverseOff = 27
    };

    template<typename... Args>
    std::string
    Colorize(const std::vector<int> &term_codes, Args... args) {
        std::stringstream ss;
        ss << "\033[" << term_codes.front();
        for (std::size_t i = 1; i < term_codes.size(); ++i) { ss << ';' << term_codes[i]; }
        ss << 'm';
        (ss << ... << args);  // https://en.cppreference.com/w/cpp/language/fold
        ss << "\033[0m";
        return ss.str();
    }

    template<typename... Args>
    inline std::string
    PrintError(Args... args) {
        return Colorize({TermCode::kTermFgRed, TermCode::kTermBold}, std::forward<Args>(args)...);
    }

    template<typename... Args>
    inline std::string
    PrintSuccess(Args... args) {
        return Colorize({TermCode::kTermFgGreen, TermCode::kTermBold}, std::forward<Args>(args)...);
    }

    template<typename... Args>
    inline std::string
    PrintWarning(Args... args) {
        return Colorize({TermCode::kTermFgYellow, TermCode::kTermBold}, std::forward<Args>(args)...);
    }

    template<typename... Args>
    inline std::string
    PrintInfo(Args... args) {
        return Colorize({TermCode::kTermFgBlue, TermCode::kTermBold}, std::forward<Args>(args)...);
    }
}  // namespace erl::common
