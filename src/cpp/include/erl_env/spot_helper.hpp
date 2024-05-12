#pragma once

#include <spot/twa/formula2bdd.hh>
#include <spot/twa/twagraph.hh>
#include <spot/misc/minato.hh>

namespace erl::env::spot_helper {

    // binary decision diagram: https://en.wikipedia.org/wiki/Binary_decision_diagram
    // spot uses BuDDy: https://buddy.sourceforge.net/manual/main.html

    /**
     * @brief Check if a state is a sink state.
     * @param aut
     * @param n
     * @return
     * @note This function is based on spot/twaalgos/neverclaim.cc:92
     */
    inline bool
    IsSink(const spot::twa_graph_ptr &aut, const uint32_t n) {
        const auto ts = aut->out(n);
        assert(ts.begin() != ts.end());
        auto it = ts.begin();
        return (it->cond == bddtrue) && (it->dst == n) && (++it == ts.end());
    }

    inline int
    GetNumberOfAtomicPropositions() {
        return bdd_varnum();
    }

    inline bdd
    FormulaToBdd(const spot::formula &f, const spot::bdd_dict_ptr &bdd_dict) {
        return spot::formula_to_bdd(f, bdd_dict, nullptr);
    }

    inline spot::formula
    BddToFormula(const bdd &b, const spot::bdd_dict_ptr &bdd_dict) {
        return spot::bdd_to_formula(b, bdd_dict);
    }

    inline bdd
    IndexToBdd(const int index) {
        return bdd_ithvar(index);
    }

    inline int
    BddToIndex(const bdd &b) {
        return bdd_var(b);
    }

    inline bdd
    ApBDDsToBddVar(const std::vector<bdd> &atomic_propositions) {
        bdd ap_vars = bddtrue;
        for (const bdd &ap: atomic_propositions) { ap_vars &= ap; }  // cppcheck-suppress useStlAlgorithm
        return ap_vars;
    }

    inline std::vector<bdd>
    BddToIrredundantSumOfProduct(const bdd &b) {
        std::vector<bdd> sops;
        spot::minato_isop isop(b);
        bdd cube;
        while ((cube = isop.next()) != bddfalse) { sops.emplace_back(cube); }
        return sops;
    }

    inline std::vector<bdd>
    BddToMinterms(const bdd &sop, const bdd &atomic_propositions) {
        std::vector<bdd> minterms;
        for (const bdd &t: minterms_of(sop, atomic_propositions)) { minterms.emplace_back(t); }  // cppcheck-suppress useStlAlgorithm
        return minterms;
    }

#define kDONT_CARE (-1)
#define kTRUE      1
#define kFALSE     0

    /**
     * @brief Convert a BDD to a vector of 0, 1, and -1 (don't care).
     * @param b
     * @return
     */
    std::vector<int>
    BddToFlags(const bdd &b);

    inline std::vector<uint32_t>
    BddToLabels(const bdd &b, const bdd &atomic_propositions) {
        // bdd -> irredundant sum of products (ISOP) -> minterms -> vectors of flags -> labels
        std::vector<uint32_t> labels;
        int varnum = bdd_varnum();
        for (const bdd &sop: BddToIrredundantSumOfProduct(b)) {
            for (const bdd &minterm: BddToMinterms(sop, atomic_propositions)) {
                uint32_t label = 0;
                // bdd_varprofile is used to count the number of appearances of each variable in a BDD.
                std::vector<int> flags = BddToFlags(minterm);
                for (int i = 0; i < varnum; ++i) {
                    if (flags[i] == kTRUE) {
                        label |= (1 << i);
                        continue;
                    }
                    ERL_ASSERTM(flags[i] != kDONT_CARE, "current bdd is not a minterm?!");
                }
                labels.emplace_back(label);
            }
        }
        return labels;
    }

    inline bdd
    LabelToBdd(uint32_t label, const std::vector<bdd> &atomic_propositions) {
        bdd result = bddtrue;
        for (std::size_t i = 0; i < atomic_propositions.size(); ++i) {
            if ((label >> i) & 1) {
                result &= atomic_propositions[i];
            } else {
                result &= !atomic_propositions[i];
            }
        }
        return result;
    }

}  // namespace erl::env::spot_helper
