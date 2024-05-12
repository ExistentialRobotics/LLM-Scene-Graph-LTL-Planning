#pragma once

#include <filesystem>
#include <fstream>
#include <functional>
#include <vector>

namespace erl::common {

    std::vector<std::vector<std::string>>
    LoadCsvFile(const char *path, char delimiter = ',');

    template<typename T>
    std::vector<std::vector<T>>
    LoadAndCastCsvFile(const char *path, std::function<T(const std::string &)> cast_func, char delimiter = ',') {
        std::ifstream ifs;
        ifs.open(path);
        if (!ifs.is_open()) { throw std::runtime_error("Fail to open file: " + std::string(path)); }

        std::vector<std::vector<T>> rows;
        std::string line;
        std::string cell;

        while (std::getline(ifs, line)) {
            std::stringstream ss(line);
            std::vector<T> row;

            while (std::getline(ss, cell, delimiter)) {
                const auto *type_of_whitespaces = "\t\n\r ";
                auto first = cell.find_first_not_of(type_of_whitespaces);
                auto last = cell.find_last_not_of(type_of_whitespaces);
                cell = cell.substr(first, (last - first + 1));
                row.push_back(cast_func(cell));
            }

            rows.push_back(row);
        }

        ifs.close();
        return rows;
    }

    template<typename T>
    void
    SaveCsvFile(const char *path, const std::vector<std::vector<T>> &rows, const char delimiter = ',') {
        std::ofstream ofs;
        ofs.open(path);
        if (!ofs.is_open()) { throw std::runtime_error("Fail to open file: " + std::string(path)); }

        for (const auto &row: rows) {
            auto iter = row.begin();

            ofs << *iter;
            ++iter;

            while (iter < row.end()) {
                ofs << delimiter << *iter;
                ++iter;
            }

            ofs << std::endl;
        }

        ofs.close();
    }

}  // namespace erl::common
