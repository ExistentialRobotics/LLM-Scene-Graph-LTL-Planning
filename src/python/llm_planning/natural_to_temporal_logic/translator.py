# code inspired from https://github.com/RoboCoachTechnologies/GPT-Synthesizer/blob/master/gpt_synthesizer/main.py
import spot
import os
import argparse

# from langchain.chains import LLMChain
# from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# from llm_planning.natural_to_temporal_logic.model import llm_init
from llm_planning.natural_to_temporal_logic.prompt import (
    get_uuid_convert_prompt,
    get_translate_prompt,
    get_syntactic_check_prompt,
)
from llm_planning.natural_to_temporal_logic.parser import (
    gpt_to_spot,
    load_building_desc,
    list_building_desc,
    find_env_elements,
    parse_syntax_error,
    save_ap_desc,
    save_ap_desc_npz,
)
from llm_planning.natural_to_temporal_logic.visualization import show_svg

spot.setup()


def get_repeated_outputs(input_dict, chain, n_repeat):
    output_dict = dict()
    most_repeated = ("", 0)

    for n in range(n_repeat):
        message = chain.invoke(input_dict)
        output = message.content
        if output in output_dict.keys():
            output_dict[output] += 1
            if output_dict[output] > most_repeated[1]:
                most_repeated = (output, output_dict[output])
        else:
            output_dict[output] = 1

    if most_repeated[1] == 0:
        most_repeated = (list(output_dict.keys())[0], 1)
        print(f"Warning: LLM produced unique outputs across all attempts!\nInput Keys: {input_dict.keys()}")

    return most_repeated[0]


def write_to_file(content, file_path: str):
    with open(file_path, "w") as file:
        file.write(str(content))


def test():
    pkg_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.abspath(os.path.join(pkg_dir, "../../../../data"))
    building_file = os.path.join(data_dir, "Collierville/building.yaml")
    instruction = "1) Water the potted plants in the living room one after the other. "
    instruction += "2) Make the beds in floor 2. "
    instruction += "3) If you pass through the kitchen, visit the garage at some point to drop the grocery bag in the car."
    output_dir = os.path.join(pkg_dir, "outputs", "test", "nl2ltl")
    os.system(f"python {__file__} --building-file {building_file} --instruction \"{instruction}\" --output-dir {output_dir}")


def main(n_chain_repeat=1, n_checker_repeat=4):
    # parse the command line arguments

    llm_model = "gpt-4"
    llm_temperature = 0.0
    llm_max_tokens = 4000
    llm_frequency_penalty = 0
    llm_presence_penalty = 0

    parser = argparse.ArgumentParser(description="Translate natural language instructions to LTL formulas.")
    parser.add_argument(
        "--building-file",
        type=str,
        required=True,
        help=f"Path to the building description file.",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        required=True,
        help=f"Natural language instruction to translate.",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=llm_model,
        help=f"Model name for the LLM. Default: {llm_model}.",
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=llm_temperature,
        help=f"Temperature parameter for the LLM. Default: {llm_temperature}.",
    )
    parser.add_argument(
        "--llm-max-tokens",
        type=int,
        default=llm_max_tokens,
        help=f"Maximum number of tokens for the LLM. Default: {llm_max_tokens}.",
    )
    parser.add_argument(
        "--llm-frequency-penalty",
        type=float,
        default=llm_frequency_penalty,
        help=f"Frequency penalty for the LLM. Default: {llm_frequency_penalty}.",
    )
    parser.add_argument(
        "--llm-presence-penalty",
        type=float,
        default=llm_presence_penalty,
        help=f"Presence penalty for the LLM. Default: {llm_presence_penalty}.",
    )
    parser.add_argument(
        "--n-chain-repeat",
        type=int,
        default=1,
        help=f"Number of times to repeat the LLM chain for each prompt. Default: {n_chain_repeat}.",
    )
    parser.add_argument(
        "--n-checker-repeat",
        type=int,
        default=4,
        help=f"Number of times to repeat the syntactic checker for each LTL formula. Default: {n_checker_repeat}.",
    )
    parser.add_argument(
        "--no-print-results",
        action="store_false",
        dest="print_results",
        help="Print the results of the translation process.",
    )
    parser.add_argument(
        "--llm-verbose",
        action="store_true",
        help="Print the outputs of the LLM chain.",
    )
    parser.add_argument(
        "--complete-automaton",
        action="store_true",
        help="Use this flag to generate a complete automaton: any possible w-word over AP is recognized.",
    )
    parser.add_argument(
        "--plot-automaton",
        action="store_true",
        help="Plot the automaton of the LTL formula.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to save the outputs. Default: current directory.",
    )
    args = parser.parse_args()
    building_file = args.building_file
    instruction = args.instruction
    llm_model = args.llm_model
    llm_temperature = args.llm_temperature
    llm_max_tokens = args.llm_max_tokens
    llm_frequency_penalty = args.llm_frequency_penalty
    llm_presence_penalty = args.llm_presence_penalty
    n_chain_repeat = args.n_chain_repeat
    n_checker_repeat = args.n_checker_repeat
    print_results = args.print_results
    verbose = args.llm_verbose
    complete_automaton = args.complete_automaton
    plot_automaton = args.plot_automaton
    output_dir = args.output_dir

    # load the scene graph
    building_desc = load_building_desc(building_file)
    building_desc_str = list_building_desc(building_desc)  # generate simplified scene graph description

    # query LLM to convert the natural language instruction to an LTL formula
    uuid_conversion_prompt = get_uuid_convert_prompt()
    llm = ChatOpenAI(
        model_name=llm_model,
        temperature=llm_temperature,
        max_tokens=llm_max_tokens,
        model_kwargs={"frequency_penalty": llm_frequency_penalty, "presence_penalty": llm_presence_penalty},
        verbose=verbose,
    )
    uuid_conversion_chain = uuid_conversion_prompt | llm

    uuid_conversion_output = get_repeated_outputs(
        input_dict={"building_desc": building_desc_str, "NL_input": instruction},
        chain=uuid_conversion_chain,
        n_repeat=n_chain_repeat,
    )
    print(uuid_conversion_output)

    translate_prompt = get_translate_prompt()
    translate_chain = translate_prompt | llm

    env_elements, env_elements_str = find_env_elements(uuid_conversion_output)

    translate_output = get_repeated_outputs(
        input_dict={"env_elements": env_elements_str, "instruction": uuid_conversion_output},
        chain=translate_chain,
        n_repeat=n_chain_repeat,
    )
    print(translate_output)

    # convert the LTL formula to a spot formula
    spot_desc, ap_dict = gpt_to_spot(translate_output)

    # perform a syntactic check on the formula
    success = False
    formula = None
    syntactic_check_prompt = get_syntactic_check_prompt()
    llm.set_verbose(True)
    syntactic_check_chain = syntactic_check_prompt | llm
    for n in range(n_checker_repeat):
        try:
            formula = spot.formula(spot_desc)
            success = True
        except SyntaxError as e:
            print("Attempt {synt_n}: Unsuccessful\nError:".format(synt_n=n))
            print(e)

            wrong_gpt_formula, error_str = parse_syntax_error(e.msg, ap_dict)

            translate_output = syntactic_check_chain.invoke(
                input=dict(instruction=uuid_conversion_output,
                syntax_error=error_str,
                wrong_LTL=wrong_gpt_formula,)
            ).content
            spot_desc, ap_dict = gpt_to_spot(translate_output)
            print(translate_output)

        if success:
            break

    if not success:
        try:
            formula = spot.formula(spot_desc)  # try one more time
        except SyntaxError as e:
            print(e)
            print("All of the syntactic checker reties were unsuccessful! Exiting the program.")
            return

    # convert the formula to an automaton
    if complete_automaton:
        automaton = spot.translate(formula, "Deterministic", "Complete")
    else:
        automaton = spot.translate(formula, "Deterministic")

    write_to_file(building_desc_str, os.path.join(output_dir, "building_desc.yaml"))
    write_to_file(instruction, os.path.join(output_dir, "NL_instructions.txt"))
    write_to_file(uuid_conversion_output, os.path.join(output_dir, "NL_instructions_uuid.txt"))
    write_to_file(env_elements_str, os.path.join(output_dir, "env_elements.txt"))
    write_to_file(translate_output, os.path.join(output_dir, "GPT_LTL_formula.txt"))
    write_to_file(spot_desc, os.path.join(output_dir, "Spot_LTL_formula.txt"))
    write_to_file(ap_dict, os.path.join(output_dir, "ap_dict.txt"))

    if print_results:
        print("\nBuilding description:")
        print(building_desc_str)

        print("\nTask description:")
        print(instruction)

        print("\nTask description with room and object UUIDs:")
        print(uuid_conversion_output)

        print("\nEnvironment elements:")
        print(env_elements_str)

        print("\nPre-fix LTL formula:")
        print(translate_output)

        print("\nSpot equivalent formula:")
        print(spot_desc)

        print("\nMapping of atomic propositions between GPT and Spot:")
        print(ap_dict)

    if plot_automaton:
        automaton_svg = automaton.show().data
        show_svg(automaton_svg)

    automaton.save(os.path.join(output_dir, "automaton.aut"))
    save_ap_desc(ap_dict, os.path.join(output_dir, "ap_desc.yaml"))
    save_ap_desc_npz(ap_dict, os.path.join(output_dir, "ap_desc.npz"))


if __name__ == "__main__":
    main()
