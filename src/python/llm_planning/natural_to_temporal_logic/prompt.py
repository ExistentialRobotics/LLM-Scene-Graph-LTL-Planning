# code inspired from https://github.com/RoboCoachTechnologies/GPT-Synthesizer/blob/master/gpt_synthesizer/prompt.py

from langchain_core.prompts import PromptTemplate


def get_translate_prompt() -> PromptTemplate:
    template = """Please help transform natural language statements into linear temporal logic (LTL) descriptions.

The LTL operators are: NEGATION, IMPLY, AND, EQUAL, UNTIL, ALWAYS, EVENTUALLY, OR.

The LTL description should follow pre-order expression.

The available actions are: enter(room_x), reach(object_y).

Some examples of natural language statements and their corresponding LTL descriptions are:

natural language: Enter room_2 if you exit room_1.
LTL:  ['IMPLY', 'NEGATION', 'enter(room_1)', 'enter(room_2)']

natural language: Enter room_2 only if you exit room_1.
LTL:  ['EQUAL', 'NEGATION', 'enter(room_1)', 'enter(room_2)']

natural language: Enter room_2 and room_1.
LTL:  ['AND', 'enter(room_2)', 'enter(room_1)']

natural language: Enter room_2, room_3, and room_1.
LTL:  ['AND', 'AND', 'enter(room_2)', 'enter(room_3)', 'enter(room_1)']

natural language: Enter room_2, room_3, room_1.
LTL:  ['AND', 'AND', 'enter(room_2)', 'enter(room_3)', 'enter(room_1)']

natural language: Clean every room_8, room_13, room_14, room_17, room_2, room_10, room_12, room_15, room_16, room_18, room_4, room_6, room_7, room_11 in the building.
LTL:  ['AND', 'AND', 'AND', 'AND', 'AND', 'AND', 'AND', 'AND', 'AND', 'AND', 'AND', 'AND', 'AND', 'enter(room_8)', 'enter(room_13)', 'enter(room_14)', 'enter(room_17)', 'enter(room_2)', 'enter(room_10)', 'enter(room_12)', 'enter(room_15)', 'enter(room_16)', 'enter(room_18)', 'enter(room_4)', 'enter(room_6)', 'enter(room_7)', 'enter(room_11)']

natural language: Going into room_1 always follows with entering room_2.
LTL:  ['ALWAYS', 'IMPLY', 'enter(room_1)', 'EVENTUALLY', 'enter(room_2)']

natural language: Maintain enter(room_1) until reach(object_2) is satisfied.
LTL:  ['ALWAYS', 'UNTIL', 'enter(room_1)', 'reach(object_2)']

natural language: Go to room_1 and always avoid both room_2 and room_3.
LTL:  ['AND', 'enter(room_1)', 'ALWAYS', 'AND', 'NEGATION', 'enter(room_2)', 'NEGATION', 'enter(room_3)']

natural language: Reach object_2 after going to room_1.
LTL:  ['AND', 'reach(object_2)', 'UNTIL', 'NEGATION', 'reach(object_2)', 'enter(room_1)']

natural language: Always if reaching object_1 happens before reaching object_3, then start object_2 and eventually cancel object_4.
LTL: ['ALWAYS', 'IMPLY' , 'AND', 'reach(object_3)', 'UNTIL', 'NEGATION', 'reach(object_3)', 'reach(object_1)', 'AND', 'reach(object_2)', 'EVENTUALLY', 'NEGATION', 'reach(object_4)']

natural language: For all time steps, until enter(room_1) and enter(room_2) is true, don't start reach(object_3).
LTL:  ['ALWAYS', 'UNTIL', 'NEGATION', 'reach(object_3)', 'AND', 'enter(room_1)', 'enter(room_2)']

natural language: If room_1 and room_2 and not room_3 or room_4, then room_5 happens eventually.
LTL:  ['IMPLY', 'AND', 'AND', 'enter(room_1)', 'enter(room_2)', 'NEGATION', 'OR', 'enter(room_3)', 'enter(room_4)', 'EVENTUALLY', 'enter(room_5)']

I will give you the list of rooms and objects in the environment, and the instruction involves entering some rooms, reaching some objects, and avoiding some rooms or objects.

Your task is to transform the natural language instruction into a LTL description with pre-order format.

Here are some examples:
Input:
    available environment elements: [room_1, room_2, room_3, room_4, room_5, object_6, object_7, object_8]
    natural language instruction: Finally reach object_7, and you have to go to room_4 ahead to enter room_1.
Output:
    LTL: ['AND', 'EVENTUALLY', 'reach(object_7)', 'UNTIL', 'NEGATION', 'enter(room_1)', 'enter(room_4)']


Input:
    available environment elements: [room_1, room_2, room_3, object_6, object_7, object_8]
    natural language instruction: Finally enter room_2, and you have to reach an object, such as object_6 or object_8, ahead to enter room_1. Remember do not enter room_3 at any time.
Output:
    LTL: ['AND', 'AND', 'EVENTUALLY', 'enter(room_2)', 'UNTIL', 'NEGATION', 'enter(room_1)', 'OR', 'OR', 'reach(object_6)', 'reach(object_7)', 'reach(object_8)', 'ALWAYS', 'NEGATION', 'enter(room_3)']


Input:
    available environment elements: [room_1, room_2, room_3, object_4, object_5 object_6, object_7, object_8]
    natural language instruction: enter room_1, then room_2 and stay there until reaching object_8, remember always keep away from object_4 and object_6.
Output:
    LTL: ['AND', 'AND', 'AND', 'enter(room_2)', 'UNTIL', 'NEGATION', 'enter(room_2)', 'enter(room_1)', 'UNTIL', 'enter(room_2)', 'reach(object_8)', 'ALWAYS', 'NEGATION', 'OR', 'reach(object_4)', 'reach(object_6)']


Input:
    available environment elements: [room_1, room_2, room_3, room_4]
    natural language instruction: Every robot should enter room_3 every time they leave room_1. After room_2, the robot should visit room_4 at some point.
Output:
    LTL: ['AND', 'ALWAYS', 'IMPLY', 'NEGATION', 'enter(room_1)', 'enter(room_3)', 'AND', 'EVENTUALLY', 'enter(room_4)', 'UNTIL', 'NEGATION', 'enter(room_4)', 'enter(room_2)']


Input:
    available environment elements: [room_1, room_2, room_3, object_6, object_7, object_8]
    natural language instruction: 1) Every robot should visit room_3 every time they leave room_2. 2) After reaching object_7, the robot should visit room_3, to transmit the collected data to the remote control. 3) The robots should avoid room_1. 4) Reach all objects.
Output:
    LTL: ['AND', 'AND', 'AND', 'ALWAYS', 'IMPLY', 'NEGATION', 'enter(room_2)', 'EVENTUALLY', 'enter(room_3)', 'AND', 'enter(room_3)', 'UNTIL', 'NEGATION', 'enter(room_3)', 'reach(object_7)', 'ALWAYS', 'NEGATION', 'enter(room_1)' 'AND', 'AND', 'reach(object_6)', 'reach(object_7)', 'reach(object_8)']


Input:
    available environment elements: [room_1, room_2, room_3, object_4, object_5, object_6]
    natural language instruction: Go to room_1, then enter room_2 and stay there until reaching object_4, and finally enter room_3. Remember always do not touch object_6.
Output:
    LTL: ['AND', 'AND', 'AND', 'AND', 'enter(room_2)', 'UNTIL', 'NEGATION', 'enter(room_2)', 'enter(room_1)', 'UNTIL', 'enter(room_2)', 'reach(object_4)', 'EVENTUALLY', 'enter(room_3)', 'ALWAYS', 'NEGATION', 'reach(object_6)']


Using the provided examples, transform the following natural language instruction into LTL specification. Do not explain the output:
Input:
    available environment elements: {env_elements}
    natural language instruction: {instruction}
Output:
LTL: """
    return PromptTemplate(template=template, input_variables=["env_elements", "instruction"])


def get_syntactic_check_prompt() -> PromptTemplate:
    template = """Please help transform natural language instructions into linear temporal logic (LTL) formulas.

The LTL operators are: NEGATION, IMPLY, AND, EQUAL, UNTIL, ALWAYS, EVENTUALLY, OR.

The LTL formula should follow pre-order expression.

The available actions are: enter(room_x), reach(object_y).

Some examples of natural language instructions and their corresponding LTL formulas are:

natural language: Enter room_2 if you exit room_1.
LTL:  ['IMPLY', 'NEGATION', 'enter(room_1)', 'enter(room_2)']

natural language: Enter room_2 only if you exit room_1.
LTL:  ['EQUAL', 'NEGATION', 'enter(room_1)', 'enter(room_2)']

natural language: Enter room_2 and room_1.
LTL:  ['AND', 'enter(room_2)', 'enter(room_1)']

natural language: Enter room_2, room_3, and room_1.
LTL:  ['AND', 'AND', 'enter(room_2)', 'enter(room_3)', 'enter(room_1)']

natural language: Enter room_2, room_3, room_1.
LTL:  ['AND', 'enter(room_2)', 'enter(room_3)', 'enter(room_1)']

natural language: Clean every room_8, room_13, room_14, room_17, room_2, room_10, room_12, room_15, room_16, room_18, room_4, room_6, room_7, room_11 in the building.
LTL:  ['AND', 'AND', 'AND', 'AND', 'AND', 'AND', 'AND', 'AND', 'AND', 'AND', 'AND', 'AND', 'AND', 'enter(room_8)', 'enter(room_13)', 'enter(room_14)', 'enter(room_17)', 'enter(room_2)', 'enter(room_10)', 'enter(room_12)', 'enter(room_15)', 'enter(room_16)', 'enter(room_18)', 'enter(room_4)', 'enter(room_6)', 'enter(room_7)', 'enter(room_11)']

natural language: Going into room_1 always follows with entering room_2.
LTL:  ['ALWAYS', 'IMPLY', 'enter(room_1)', 'EVENTUALLY', 'enter(room_2)']

natural language: Maintain enter(room_1) until reach(object_2) is satisfied.
LTL:  ['ALWAYS', 'UNTIL', 'enter(room_1)', 'reach(object_2)']

natural language: Go to room_1 and always avoid both room_2 and room_3.
LTL:  ['AND', 'enter(room_1)', 'ALWAYS', 'AND', 'NEGATION', 'enter(room_2)', 'NEGATION', 'enter(room_3)']

natural language: Reach object_2 after going to room_1.
LTL:  ['AND', 'reach(object_2)', 'UNTIL', 'NEGATION', 'reach(object_2)', 'enter(room_1)']

natural language: Always if reaching object_1 happens before reaching object_3, then start object_2 and eventually cancel object_4.
LTL: ['ALWAYS', 'IMPLY' , 'AND', 'reach(object_3)', 'UNTIL', 'NEGATION', 'reach(object_3)', 'reach(object_1)', 'AND', 'reach(object_2)', 'EVENTUALLY', 'NEGATION', 'reach(object_4)']

natural language: For all time steps, until enter(room_1) and enter(room_2) is true, don't start reach(object_3).
LTL:  ['ALWAYS', 'UNTIL', 'NEGATION', 'reach(object_3)', 'AND', 'enter(room_1)', 'enter(room_2)']

natural language: If room_1 and room_2 and not room_3 or room_4, then room_5 happens eventually.
LTL:  ['IMPLY', 'AND', 'AND', 'enter(room_1)', 'enter(room_2)', 'NEGATION', 'OR', 'enter(room_3)', 'enter(room_4)', 'EVENTUALLY', 'enter(room_5)']

natural language: Finally reach object_7, and you have to go to room_4 ahead to enter room_1.
LTL: ['AND', 'EVENTUALLY', 'reach(object_7)', 'UNTIL', 'NEGATION', 'enter(room_1)', 'enter(room_4)']

natural language: Finally enter room_2, and you have to reach object_6 or object_8, ahead to enter room_1. Remember do not enter room_3 at any time.
LTL: ['AND', 'AND', 'EVENTUALLY', 'enter(room_2)', 'UNTIL', 'NEGATION', 'enter(room_1)', 'OR', 'reach(object_6)', 'reach(object_8)', 'ALWAYS', 'NEGATION', 'enter(room_3)']

natural language: enter room_1, then room_2 and stay there until reaching object_8, remember always keep away from object_4 and object_6.
LTL: ['AND', 'AND', 'AND', 'enter(room_2)', 'UNTIL', 'NEGATION', 'enter(room_2)', 'enter(room_1)', 'UNTIL', 'enter(room_2)', 'reach(object_8)', 'ALWAYS', 'NEGATION', 'OR', 'reach(object_4)', 'reach(object_6)']

natural language: Every robot should enter room_3 every time they leave room_1. After room_2, the robot should visit room_4 at some point.
LTL: ['AND', 'ALWAYS', 'IMPLY', 'NEGATION', 'enter(room_1)', 'enter(room_3)', 'AND', 'EVENTUALLY', 'enter(room_4)', 'UNTIL', 'NEGATION', 'enter(room_4)', 'enter(room_2)']

natural language: 1) Every robot should visit room_3 every time they leave room_2. 2) After reaching object_7, the robot should visit room_3, to transmit the collected data to the remote control. 3) The robots should avoid room_1. 4) Reach object_6, object_7, and object_8.
LTL: ['AND', 'AND', 'AND', 'ALWAYS', 'IMPLY', 'NEGATION', 'enter(room_2)', 'EVENTUALLY', 'enter(room_3)', 'AND', 'enter(room_3)', 'UNTIL', 'NEGATION', 'enter(room_3)', 'reach(object_7)', 'ALWAYS', 'NEGATION', 'enter(room_1)' 'AND', 'AND', 'reach(object_6)', 'reach(object_7)', 'reach(object_8)']

natural language: Go to room_1, then enter room_2 and stay there until reaching object_4, and finally enter room_3. Remember always do not touch object_6.
LTL: ['AND', 'AND', 'AND', 'AND', 'enter(room_2)', 'UNTIL', 'NEGATION', 'enter(room_2)', 'enter(room_1)', 'UNTIL', 'enter(room_2)', 'reach(object_4)', 'EVENTUALLY', 'enter(room_3)', 'ALWAYS', 'NEGATION', 'reach(object_6)']


Trained by the above examples, an AI has generated a syntactically INCORRECT LTL formula for the following natural language instruction:
natural language: {instruction}
Incorrect LTL: {wrong_LTL}

The incorrect part of the LTL formula is shown with '--> INCORRECT', meaning that this part is causing the error and needs to be modified.

More specifically, here is a description of the error caused by the incorrect part:
{syntax_error}

Generate a syntactically correct revision of the LTL formula similar to the examples provided above.

Pay attention to the number of elements that each LTL operator requires.

For example, 'AND', 'OR', 'EQUAL', 'IMPLY', 'UNTIL' operators take two inputs: ['AND', 'enter(room_1)', 'reach(object_2)']
On the other hand, 'NEGATION', 'ALWAYS', 'EVENTUALLY' only take a single input: ['EVENTUALLY', 'reach(object_2)']

natural language: {instruction}
Corrected LTL: """
    return PromptTemplate(template=template, input_variables=["instruction", "syntax_error", "wrong_LTL"])


def get_uuid_convert_prompt() -> PromptTemplate:
    template = """Your task is to convert the room and object names in a text to their unique id based on a given description of the building.

Use the room and object IDs provided in the building description for conversion.

The conversion should take the context of each sentence into account, so that rooms or objects with similar names but different locations can be correctly distinguished.

Here are a few examples:

Building description:
- bathroom_2:
        - floor: 0
        - id: room_2
        - objects:
            ['sink_27': 'object_27', 'toilet_41': 'object_41', 'potted plant_51': 'object_51']
- corridor_9:
        - floor: 0
        - id: room_9
        - objects:
            []
- empty_room_14:
        - floor: 0
        - id: room_14
        - objects:
            []
- lobby_17:
        - floor: 0
        - id: room_17
        - objects:
            []
- staircase_18:
        - floor: 0
        - id: room_18
        - objects:
            []
- corridor_10:
        - floor: 1
        - id: room_10
        - objects:
            ['chair_46': 'object_46', 'potted plant_49': 'object_49', 'potted plant_50': 'object_50']
- dining_room_13:
        - floor: 1
        - id: room_13
        - objects:
            ['bowl_39': 'object_39', 'chair_43': 'object_43', 'chair_44': 'object_44', 'chair_45': 'object_45', 'dining table_55': 'object_55', 'dining table_56': 'object_56', 'dining table_57': 'object_57', 'dining table_58': 'object_58']
- kitchen_15:
        - floor: 1
        - id: room_15
        - objects:
            ['microwave_21': 'object_21', 'oven_22': 'object_22', 'sink_23': 'object_23', 'refrigerator_28': 'object_28', 'bottle_31': 'object_31', 'bottle_32': 'object_32', 'bowl_37': 'object_37', 'bowl_38': 'object_38']
- living_room_16:
        - floor: 1
        - id: room_16
        - objects:
            ['teddy bear_36': 'object_36', 'couch_48': 'object_48', 'tv_60': 'object_60', 'tv_61': 'object_61']
Input text: Take the teddy bear in the living room, then pick the bottle in the kitchen. Always avoid the corridor in floor 0.
Output text: Take object_36 in room_16, then pick object_31 in room_15. Always avoid the corridor in room_9.


Building description:
- bathroom_4:
        - floor: 2
        - id: room_4
        - objects:
            ['sink_24': 'object_24', 'toilet_40': 'object_40']
- bathroom_6:
        - floor: 2
        - id: room_6
        - objects:
            ['sink_25': 'object_25', 'sink_26': 'object_26', 'toilet_42': 'object_42']
- bedroom_7:
        - floor: 2
        - id: room_7
        - objects:
            ['book_30': 'object_30', 'vase_34': 'object_34', 'chair_47': 'object_47', 'bed_54': 'object_54']
- bedroom_8:
        - floor: 2
        - id: room_8
        - objects:
            ['book_29': 'object_29', 'bottle_33': 'object_33', 'teddy bear_35': 'object_35', 'bed_52': 'object_52', 'bed_53': 'object_53', 'tv_59': 'object_59']
- corridor_12:
        - floor: 2
        - id: room_12
        - objects:
            []
- utility_room_20:
        - floor: 2
        - id: room_20
        - objects:
            []
Input text: Go to the bedroom with a vase, and always avoid going to the utility room and the bathroom with two sinks.
Output text: Go to room_7, and always avoid going to the room_20 and room_6.


Building description:
- corridor_10:
        - floor: 1
        - id: room_10
        - objects:
            ['chair_46': 'object_46', 'potted plant_49': 'object_49', 'potted plant_50': 'object_50']
- dining_room_13:
        - floor: 1
        - id: room_13
        - objects:
            ['bowl_39': 'object_39', 'chair_43': 'object_43', 'chair_44': 'object_44', 'chair_45': 'object_45', 'dining table_55': 'object_55', 'dining table_56': 'object_56', 'dining table_57': 'object_57', 'dining table_58': 'object_58']
- kitchen_15:
        - floor: 1
        - id: room_15
        - objects:
            ['microwave_21': 'object_21', 'oven_22': 'object_22', 'sink_23': 'object_23', 'refrigerator_28': 'object_28', 'bottle_31': 'object_31', 'bottle_32': 'object_32', 'bowl_37': 'object_37', 'bowl_38': 'object_38']
- living_room_16:
        - floor: 1
        - id: room_16
        - objects:
            ['teddy bear_36': 'object_36', 'couch_48': 'object_48', 'tv_60': 'object_60', 'tv_61': 'object_61']
- staircase_19:
        - floor: 1
        - id: room_19
        - objects:
            []
- bathroom_4:
        - floor: 2
        - id: room_4
        - objects:
            ['sink_24': 'object_24', 'toilet_40': 'object_40']
- bathroom_6:
        - floor: 2
        - id: room_6
        - objects:
            ['sink_25': 'object_25', 'sink_26': 'object_26', 'toilet_42': 'object_42']
Input text: Exit the kitchen through the dining room and go to the living room but avoid reaching any objects in the room.
Output text: Exit room_15 through room_13 and go to room_16 but avoid reaching object_36, object_48, object_60, and object_61.


Building description:
- corridor_9:
        - floor: 0
        - id: room_9
        - objects:
            []
- lobby_17:
        - floor: 0
        - id: room_17
        - objects:
            []
- staircase_18:
        - floor: 0
        - id: room_18
        - objects:
            []
- corridor_10:
        - floor: 1
        - id: room_10
        - objects:
            ['chair_46': 'object_46', 'potted plant_49': 'object_49', 'potted plant_50': 'object_50']
- dining_room_13:
        - floor: 1
        - id: room_13
        - objects:
            ['bowl_39': 'object_39', 'chair_43': 'object_43', 'chair_44': 'object_44', 'chair_45': 'object_45', 'dining table_55': 'object_55', 'dining table_56': 'object_56', 'dining table_57': 'object_57', 'dining table_58': 'object_58']
- kitchen_15:
        - floor: 1
        - id: room_15
        - objects:
            ['microwave_21': 'object_21', 'oven_22': 'object_22', 'sink_23': 'object_23', 'refrigerator_28': 'object_28', 'bottle_31': 'object_31', 'bottle_32': 'object_32', 'bowl_37': 'object_37', 'bowl_38': 'object_38']
- living_room_16:
        - floor: 1
        - id: room_16
        - objects:
            ['teddy bear_36': 'object_36', 'couch_48': 'object_48', 'tv_60': 'object_60', 'tv_61': 'object_61']
Input text: Pick the food from the refrigerator, and put it in the microwave. Then wait in the corridor in floor 1.
Output text: Pick object_38 from object_28, and put it in object_21. Then wait in room_10.


Building description:
- bathroom_4:
        - floor: 2
        - id: room_4
        - objects:
            ['sink_24': 'object_24', 'toilet_40': 'object_40']
- bathroom_6:
        - floor: 2
        - id: room_6
        - objects:
            ['sink_25': 'object_25', 'sink_26': 'object_26', 'toilet_42': 'object_42']
- bedroom_7:
        - floor: 2
        - id: room_7
        - objects:
            ['book_30': 'object_30', 'vase_34': 'object_34', 'chair_47': 'object_47', 'bed_54': 'object_54']
- bedroom_8:
        - floor: 2
        - id: room_8
        - objects:
            ['book_29': 'object_29', 'bottle_33': 'object_33', 'teddy bear_35': 'object_35', 'bed_52': 'object_52', 'bed_53': 'object_53', 'tv_59': 'object_59']
Input text: One of the bathrooms contains two sinks. Also, one of the bedrooms has a TV, while the other has a vase.
Output text: room_6 contains object_25 and object_26. Also, room_8 has object_59, while room_7 has object_34.


Building description:
- corridor_12:
        - floor: 2
        - id: room_12
        - objects:
            []
- corridor_10:
        - floor: 1
        - id: room_10
        - objects:
            ['chair_46': 'object_46', 'potted plant_49': 'object_49', 'potted plant_50': 'object_50']
- corridor_9:
        - floor: 0
        - id: room_9
        - objects:
            []
Input text: The corridor with objects is in between the corridor in floor 0 and the corridor in floor 2.
Output text: room_10 is in between room_9 and room_12.


Using the provided examples, convert the room and objects in the following text into their unique IDs.

Building description:
{building_desc}
Input text: {NL_input}
Output text: """
    return PromptTemplate(template=template, input_variables=["building_desc", "NL_input"])
