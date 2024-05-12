#include "erl_common/csv.hpp"

namespace erl::common {

    std::vector<std::vector<std::string>>
    LoadCsvFile(const char* path, char delimiter) {
        std::ifstream ifs;
        ifs.open(path);
        if (!ifs.is_open()) { throw std::runtime_error("Fail to open file: " + std::string(path)); }

        std::vector<std::vector<std::string>> rows;
        std::string line;
        std::string cell;

        while (std::getline(ifs, line)) {
            std::stringstream ss(line);
            std::vector<std::string> row;

            while (std::getline(ss, cell, delimiter)) {
                const auto* type_of_whitespaces = "\t\n\r ";
                auto first = cell.find_first_not_of(type_of_whitespaces);
                auto last = cell.find_last_not_of(type_of_whitespaces);
                cell = cell.substr(first, last - first + 1);
                row.push_back(cell);
            }

            rows.push_back(row);
        }

        ifs.close();
        return rows;
    }
}  // namespace erl::common
