# code inspired from https://github.com/RoboCoachTechnologies/GPT-Synthesizer/blob/master/gpt_synthesizer/main.py
import spot

from llm_planning.natural_to_temporal_logic.parser import gpt_to_spot
from llm_planning.natural_to_temporal_logic.visualization import show_svg
from llm_planning.natural_to_temporal_logic.parser import save_ap_desc, save_ap_desc_npz

spot.setup()


def main(verbose=True):
    instruction = "Visit the bathroom in floor 0, while avoiding the sink. Then go to the dining room and sit on one of the chairs. Always avoid the living room and the staircase next to it."
    uuid_instruction = "Visit room_2 in floor 0, while avoiding object_27. Then go to room_13 and sit on one of object_43, object_44, or object_45. Always avoid room_16 and the room_19 next to it."
    translate_output = "['AND', 'EVENTUALLY', 'AND', 'AND', 'enter(room_2)', 'ALWAYS', 'NEGATION', 'reach(object_27)', 'EVENTUALLY', 'AND', 'enter(room_13)', 'EVENTUALLY', 'OR', 'OR', 'reach(object_43)', 'reach(object_44)', 'reach(object_45)', 'ALWAYS', 'AND', 'NEGATION', 'enter(room_16)', 'NEGATION', 'enter(room_19)']"
    spot_desc, ap_dict = gpt_to_spot(translate_output)
    formula = spot.formula(spot_desc)

    automaton = spot.translate(formula)
    automaton_svg = automaton.show().data

    if verbose:
        print("\nTask description:")
        print(instruction)

        print("\nTask description with UUID:")
        print(uuid_instruction)

        print("\nSpot equivalent formula:")
        print(spot_desc)

        print("\nMapping of atomic propositions between GPT and Spot:")
        print(ap_dict)

    show_svg(automaton_svg)

    automaton.save("automaton.aut")
    save_ap_desc(ap_dict)
    save_ap_desc_npz(ap_dict)

    return


if __name__ == "__main__":
    main()
