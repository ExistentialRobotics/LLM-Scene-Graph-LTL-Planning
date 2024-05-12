# code inspired from https://github.com/RoboCoachTechnologies/GPT-Synthesizer/blob/master/gpt_synthesizer/prompt.py
# prompts inspired from https://github.com/yongchao98/AutoTAMP/tree/main

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
LTL:  ['AND", 'AND', 'enter(room_2)', 'enter(room_3)', 'enter(room_1)']

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
LTL:  ['AND', 'enter(room_2)', 'enter(room_3)', 'enter(room_1)']

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


# NL2STL_prompt="""Try to transform the following natural language statements into linear temporal logic (LTL) descriptions, the operators in the linear temporal logic are:
# negation, imply, and, equal, until, always, eventually, or.
#
# The examples are as following:
# natural language: If ( prop_2 ) is equivalent to ( prop_3 ) and also ( prop_4 ) , then this scenario is equivalent to ( prop_1 ) .
# STL: ( ( ( prop_2 equal prop_3 ) and prop_4 ) equal prop_1 )
#
# natural language: For all times either ( prop_1 ) or ( prop_2 ) should be detected , or else ( prop_3 ) .
# STL: ( ( always ( prop_1 or prop_2 ) ) or prop_3 )
#
# natural language: It is required that at some point the scenario in which ( prop_3 ) is equivalent to the scenario in which ( prop_2 ) happens , and only then ( prop_1 ) .
# STL: ( ( eventually ( prop_3 equal prop_2 ) ) imply prop_1 )
#
# natural language: In case that at some point ( prop_4 ) or ( prop_2 ) or ( prop_3 ) is detected and continued until then at some other point ( prop_1 ) should be detected as well .
# STL: ( ( ( prop_4 or prop_2 ) or prop_3 ) until prop_1 )
#
# natural language: ( prop_2 ) should happen and hold until at a certain time point the scenario that ( prop_3 ) should happen then ( prop_1 ) , or else ( prop_4 ) .
# STL: ( ( ( prop_2 until prop_3 ) imply prop_1 ) or prop_4 )
#
# natural language: For each time instant if it is not the case that ( prop_1 ) or ( prop_3 ) then ( prop_2 ) .
# STL: ( ( always ( negation ( prop_1 or prop_3 ) ) ) imply prop_2 )
# ------------------------------------
# natural language: In case that ( prop_3 ) continues to happen until at some point during the first 391 to 525 time units that ( prop_2 ) happens , as well as ( prop_1 ) , and ( prop_4 ) then .
# STL: ( ( ( prop_3 until [391,525] prop_2 ) and prop_1 ) and prop_4 )
#
# natural language: If finally that ( prop_1 ) is not detected then ( prop_2 ) , then ( prop_3 ) .
# STL: ( ( ( finally ( negation prop_1 ) ) imply prop_2 ) imply prop_3 )
#
# natural language: It is not the case that ( prop_1 ) if and only if ( prop_2 ) is true , the above scenario will hold until ( prop_3 ) will be detected at some time point during the next 394 to 530 time units .
# STL: ( ( negation ( prop_1 equal prop_2 ) ) until [394,530] prop_3 )
#
# natural language:  If at some point ( prop_1 ) or ( prop_3 ) then ( prop_4 ) happens and this scenario will hold until at some other point during the 193 to 266 time units ( prop_2 ) is detected .
# STL: ( ( ( prop_1 or prop_3 ) imply prop_4 ) until [193,266] prop_2 )
#
# natural language:  It is not the case that ( prop_1 ) happens and continues to happen until at some point during the 77 to 432 time units ( prop_2 ) is detected , and ( prop_3 ) .
# STL: ( ( negation ( prop_1 until [77,432] prop_2 ) ) and prop_3 )
#
# natural language: ( prop_3 ) happens until a time in the next 5 to 12 units that ( prop_4 ) does not happen .
# STL: ( prop_3 until [5,12] ( negation prop_4 ) )
#
# natural language: The time that ( prop_3 ) happens is when ( prop_1 ) happens , and vice versa .
# STL: ( prop_3 equal prop_1 )
#
# natural language: It is required that both ( prop_2 ) and ( prop_4 ) happen at the same time, or else ( prop_3 ) happens and continues until ( prop_1 ) happens.
# STL:  ( ( prop_2 and prop_4 ) or ( prop_3 until prop_1 ) )
#
# natural language: ( prop_3 ) happens and continues until at some point during the 500 to 903 time units ( prop_1 ) happens , and in the same time ( prop_2 ) does not happen .
# STL:  ( ( prop_3 until [500,903] prop_1 ) and ( negation prop_2 ) )
#
# natural language: For each time instant in the next 107 to 513 time units ( prop_1 ) is true , or else ( prop_3 ) happens and ( prop_2 ) happens at the same time.
# STL:  ( ( globally [107,513] prop_1 ) or ( prop_3 and prop_2 ) )
#
# natural language: ( prop_1 ) or ( prop_2 ) happens and continues until at some point during the 142 to 365 time units ( prop_4 ) happens and ( prop_3 ) happens at the same time .
# STL:  ( ( prop_1 or prop_2 ) until [142,365] ( prop_4 and prop_3 ) )
#
# natural language: It is always the case that everytime when ( prop_1 ) and ( prop_2 ) then all of the following conditions hold : for each time point during the subsequent 4 to 47 time units ( prop_3 ) .
# STL:  globally ( ( prop_1 and prop_2 ) imply ( globally [4,47] prop_3 ) )
#
# natural language: at some time ( prop_1 ) and when possible ( prop_2 )
# STL:  finally ( prop_1 and finally prop_2 )
#
# natural language: at some time ( prop_1 ) and never ( prop_2 )
# STL:  ( finally prop_1 ) and ( globally ( negation prop_2 ) )
#
# natural language: when possible ( prop_1 ) and at some time ( prop_2 )
# STL:  finally ( prop_1 and ( finally prop_2 ) )
#
# natural language: never ( prop_1 ) or whenever ( prop_2 )
# STL:  ( globally ( ( negation prop_1 ) ) or ( finally prop_2 ) )
#
# natural language: when possible ( prop_1 ) and don't ( prop_2 )
# STL:  finally ( prop_1 and ( negation prop_2 ) )
#
# natural language: ( prop_1 ) or at any time ( prop_2 )
# STL:  ( prop_1 or ( finally prop_2 ) )
#
# natural language: Globally , if ( prop_1 ) then the following condition is true : at the same time the event that ( prop_2 ) ought to be detected and ( prop_3 ) .
# STL:  globally ( prop_1 imply ( prop_2 and prop_3 ) )
#
# natural language: at some time ( prop_1 ) or ( prop_2 )
# STL:  finally ( prop_1 or prop_2 )
#
# natural language: ( prop_1 ) and at some time ( prop_2 )
# STL:  ( prop_1 and ( finally prop_2 ) )
#
# natural language: whenever ( prop_1 ) and at any time ( prop_2 )
# STL:  ( ( finally prop_1 ) and ( finally prop_2 ) )
#
# natural language: whenever ( prop_1 ) or forever ( prop_2 )
# STL:  ( ( finally prop_1 ) or ( globally prop_2 ) )
#
# natural language: At the random time between 0 to 30 (prop_1) for 5 seconds, and at the random time between 0 to 30 (prop_2) for 5 seconds, and at the random time between 0 to 30 (prop_3) for 5 seconds.
# STL: ( ( ( finally [0,30] ( globally [0,5] prop_1 ) ) and ( finally [0,30] ( globally [0,5] prop_2 ) ) ) and ( finally [0,30] ( globally [0,5] prop_3 ) ) )
#
# natural language: At sometime between 1 to 25 stay at prop_1 for five seconds and always stay away from the prop_2 and prop_3 during 0 to 30 seconds.
# STL: ( ( finally [1,25] ( globally [0,5] prop_1 ) ) and ( globally [0,30] ( negation prop_2 and negation prop_3 ) ) )
#
# natural language: Always (prop_1) if not (prop_2).
# STL: ( ( negation prop_2 ) imply ( globally prop_1 ) )
#
# natural language: Always (prop_1) only if not (prop_2).
# STL: ( (globally prop_1 ) imply ( negation prop_2 ) )
#
# natural language: Avoid (prop_1) until (prop_2) lasts for 20 steps.
# STL: ( ( negation prop_1 ) until ( globally [0, 20] prop_2 ) )
#
# natural language: Wait till (prop_1) is complete, then, (prop_2) for next 20 units.
# STL: ( ( finally prop_1 ) imply ( globally [0, 20] prop_2 ) )
#
# natural language: ( prop_1 ) won't happen only if ( prop_2 )
# STL: ( ( negation prop_1 ) imply prop_2 )
#
# natural language: only under the case of ( prop_2 ), will ( prop_1 ) not happen
# STL: ( ( negation prop_1 ) imply prop_2 )
#
# natural language: ( prop_1 ) always follows with ( prop_2 ).
# STL: ( globally ( prop_1 imply ( finally prop_2 ) ) )
#
# natural language: Maintain ( prop_1 ) until ( prop_2 ) is satisfied.
# STL: globally ( prop_1 until prop_2 )
#
# natural language: Go to ( prop_1 ) and always avoid both ( prop_2 ) and ( prop_3 ).
# STL: ( ( finally prop_1 ) and globally ( ( negation prop_2 ) and ( negation prop_3 ) ) )
#
# natural language: If ( prop_1 ) happens before ( prop_3 ) then start ( prop_2 ) and cancel ( prop_4 ) anytime within 0 to 10 timesteps.
# STL: globally ( ( prop_1 and ( finally prop_3 ) ) imply ( prop_2 and ( finally [0,10] ( negation prop_4 ) ) ) )
#
# natural language: For time steps between 0 and 20, until (prop_1) and (prop_2) is true, don’t start ( prop_3 ).
# STL: ( globally [0,20] ( ( negation prop_3 ) until ( prop_1 and prop_2 ) ) )
#
# natural language: If prop_1 and prop_2 and not prop_3 or prop_4, then prop_5 happens after 10 timesteps.
# STL: ( ( prop_1 and prop_2 ) and ( negation ( prop_3 or prop_4 ) ) ) imply ( finally [10,infinite] prop_5 )
#
# natural language: prop_1 for the first 5 timesteps then prop_2 thereafter.
# STL: ( ( globally [0,5] prop_1 ) and ( globally [5,infinite] prop_2 ) )
#
# natural language: prop_1 and prop_2 and prop_3 and prop_4 and prop_5
# STL: ( ( ( ( prop_1 and prop_2 ) and prop_3 ) and prop_4 ) and prop_5 )
#
# natural language: prop_1 and prop_2 and prop_3
# STL: ( ( prop_1 and prop_2 ) and prop_3 )
#
# natural language: {task}
# STL:"""
#
#
#
# NL2STL_prompt="""Try to transform the following natural languages into signal temporal logics, the operators in the signal temporal logic are:
# negation, imply, and, equal, until, globally, finally, or .
#
# The signal temporal logics are prefix expressions.
#
# The examples are as following:
# natural language: If ( prop_3 ) then implies ( prop_4 ), this condition should continue to happen until at some point within the next 450 to 942 time units , after that ( prop_2 ) , or ( prop_1 ) .
# STL:  ['or', 'until [450,942]', 'imply', 'prop_3', 'prop_4', 'prop_2', 'prop_1']
#
# natural language: The time that ( prop_3 ) happens is when ( prop_1 ) happens , and vice versa .
# STL:  ['equal', 'prop_3', 'prop_1']
#
# natural language: ( prop_1 ) or ( prop_2 ) happens and continues until at some point during the 142 to 365 time units ( prop_4 ) happens and ( prop_3 ) happens at the same time .
# STL:  ['until [142,365]', 'or', 'prop_1', 'prop_2', 'and', 'prop_4', 'prop_3']
#
# natural language: ( prop_1 ) should not happen and ( prop_2 ) should happen at the same time , and the above scenario is equivalent to the case that at some point during the 230 to 280 time units ( prop_3 ) happens .
# STL:  ['equal', 'and', 'negation', 'prop_1', 'prop_2', 'finally [230,280]', 'prop_3']
#
# natural language: In the following 10 time steps , the ( prop_1 ) should always happen , and in the meantime , ( prop_2 ) should happen at least once .
# STL:  ['and', 'globally [0,10]', 'prop_1', 'finally', 'prop_2']
#
# natural language: ( prop_1 ) should not happen if ( prop_2 ) does not happen , and ( prop_3 ) should also be true all the time .
# STL:  ['and', 'imply', 'negation', 'prop_2', 'negation', 'prop_1', 'globally', 'prop_3']
#
# natural language: If ( prop_1 ) and ( prop_2 ), then ( prop_3 ) until ( prop_4 ) does not happen , and ( prop_5 ) until ( prop_6 ) does not happen .
# STL:  ['and', 'imply', 'and', 'prop_1', 'prop_2', 'until', 'prop_3', 'negation', 'prop_4', 'until', 'prop_5', 'negation', 'prop_6']
#
# natural language: For each time instant in the next 0 to 120 units, do ( prop_1 ) if ( prop_2 ) , and if possible, ( prop_4 ) .
# STL:  ['and', 'globally [0,120]', 'imply', 'prop_2', 'prop_1', 'prop_4']
#
# natural language: In the next 0 to 5 time units , do the ( prop_1 ) , but in the next 3 to 4 time units , ( prop_2 ) should not happen .
# STL:  ['and', 'globally [0,5]', 'prop_1', 'globally [3,4]', 'negation', 'prop_2']
#
# natural language: While (prop_1) , do (prop_2) , and when (prop_3) , stop (prop_2) .
# STL: ['and', 'imply', 'prop_1', 'prop_2', 'imply', 'prop_3', 'negation', 'prop_2']
#
# natural language: If (prop_1) happens, then some time after the next 300 time steps (prop_2) should happen.
# STL:  ['imply', 'prop_1', 'finally [300, infinite]', 'prop_2']
#
# natural language: If (prop_1) happens, then for all time afterward (prop_2) holds and if, in addition, if (prop_3) occurs, then (prop_4) eventually occurs in the next 10 time units.
# STL:  ['imply', 'prop_1', 'and', 'globally', 'prop_2', 'imply', 'prop_3', 'finally [0, 10]', 'prop_4']
#
# natural language: If (prop_1), don't (prop_2), instead keep (prop_3) until (prop_4).
# STL: ['imply', 'prop_1', 'and', 'negation', 'prop_2', 'until', 'prop_3', 'prop_4']
#
# natural language: If (prop_4), then make sure any of the following happens: (prop_1), (prop_2) or (prop_3).
# STL:  ['imply', 'prop_4', 'or', 'or', 'prop_1', 'prop_2', 'prop_3']
#
# natural language: Always make (prop_1) happen in the next 999 time units if (prop_2) in the next 500 time instants.
# STL:  ['imply', 'finally [0, 500]', 'prop_2', 'finally [0, 999]', 'prop_1']
#
# natural language: If (prop_1) happens, then keep (prop_2) to be true until (prop_3) in the next 300 time units, otherwise, if (prop_4) then (prop_2) and if (prop_5) then (prop_6).
# STL:  ['imply', 'prop_1', 'or', 'until [0, 300]', 'prop_2', 'prop_3', 'and', 'imply', 'prop_4', 'prop_2', 'imply', 'prop_5', 'prop_6']
#
# natural language: Stay (prop_1) for 354 timesteps, and if (prop_2) happens, then first keep (prop_3) and then let (prop_4) happen at some point during 521 to 996 time steps.
# STL: ['and', 'globally [0, 354]', 'prop_1', 'imply', 'prop_2', 'and', 'globally [0, 521]', 'prop_3', 'finally [521, 996]', 'prop_4']
#
# natural language: Manage to achieve (prop_1) in the next 1000 time steps, and if (prop_2) happens in this process, keep (prop_3) until (prop_4) for 500 time units.
# STL: ['and', 'finally [0, 1000]', 'prop_1', 'imply', 'prop_2', 'until [0, 500]', 'prop_3', 'prop_4']
#
# natural language: As long as (prop_1), make sure to maintain (prop_2) .
# STL:  ['imply', 'prop_1', 'prop_2']
#
# natural language: Do (prop_1) until (prop_3), but once (prop_4) occurs then immediately (prop_2) .
# STL:  ['and', 'until', 'prop_3', 'prop_1', 'imply', 'prop_4', 'prop_2']
#
# natural language: If you do (prop_1) and observe (prop_2), then you should not do (prop_3) .
# STL:  ['imply', 'and', 'prop_1', 'prop_2', 'negation', 'prop_3']
#
# natural language: The time that ( prop_3 ) happens is when ( prop_1 ) happens , and vice versa .
# STL: ['equal', 'prop_1', 'prop_2']
#
# natural language: It is required that both ( prop_2 ) and ( prop_4 ) happen at the same time, or else ( prop_3 ) happens and continues until ( prop_1 ) happens.
# STL:  ['or', 'and', 'prop_2', 'prop_4', 'until', 'prop_3', 'prop_1']
#
# natural language: ( prop_3 ) happens and continues until at some point during the 500 to 903 time units ( prop_1 ) happens , and in the same time ( prop_2 ) does not happen .
# STL:  ['and', 'until [500,903]', 'prop_3', 'prop_1', 'negation', 'prop_2']
#
# natural language: For each time instant in the next 107 to 513 time units ( prop_1 ) is true , or else ( prop_3 ) happens and ( prop_2 ) happens at the same time.
# STL: ['or', 'globally [107, 513]', 'prop_1', 'and', 'prop_3', 'prop_2']
#
# natural language: ( prop_1 ) or ( prop_2 ) happens and continues until at some point during the 142 to 365 time units ( prop_4 ) happens and ( prop_3 ) happens at the same time .
# STL:  ['until [142,365]', 'or', 'prop_1', 'prop_2', 'and', 'prop_4', 'prop_3']
#
# natural language: It is always the case that everytime when ( prop_1 ) and ( prop_2 ) then all of the following conditions hold : for each time point during the subsequent 4 to 47 time units ( prop_3 ) .
# STL:  ['imply', 'and', 'prop_1', 'prop_2', 'globally [4,47]', 'prop_3']
#
# natural language: at some time ( prop_1 ) and when possible ( prop_2 )
# STL:  ['and', 'prop_1', 'finally', 'prop_2']
#
# natural language: at some time ( prop_1 ) and never ( prop_2 )
# STL:  ['and', 'finally', 'prop_1', 'globally', 'negation', 'prop_2']
#
# natural language: when possible ( prop_1 ) and at some time ( prop_2 )
# STL:  ['finally', 'and', 'prop_1', 'finally', 'prop_2']
#
# natural language: never ( prop_1 ) or whenever ( prop_2 )
# STL:  ['or', 'globally', 'negation', 'prop_1', 'finally', 'prop_2']
#
# natural language: when possible ( prop_1 ) and don't ( prop_2 )
# STL:  ['finally', 'and', 'prop_1', 'negation', 'prop_2']
#
# natural language: ( prop_1 ) or at any time ( prop_2 )
# STL:  ['or', 'prop_1', 'finally', 'prop_2']
#
# natural language: Globally , if ( prop_1 ) then the following condition is true : at the same time the event that ( prop_2 ) ought to be detected and ( prop_3 ) .
# STL:  ['globally', 'imply', 'prop_1', 'and', 'prop_2', 'prop_3']
#
# natural language: at some time ( prop_1 ) or ( prop_2 )
# STL:  ['finally', 'or', 'prop_1', 'prop_2']
#
# natural language: ( prop_1 ) and at some time ( prop_2 )
# STL:  ['and', 'prop_1', 'finally', 'prop_2']
#
# natural language: whenever ( prop_1 ) and at any time ( prop_2 )
# STL:  ['and', 'finally', 'prop_1', 'finally', 'prop_2']
#
# natural language: whenever ( prop_1 ) or forever ( prop_2 )
# STL: ['or', 'finally', 'prop_1', 'globally', 'prop_2']
#
# natural language: At the random time between 0 to 30 (prop_1) for 5 seconds, and at the random time between 0 to 30 (prop_2) for 5 seconds, and at the random time between 0 to 30 (prop_3) for 5 seconds.
# STL:  ['and', 'and', 'finally [0,30]', 'globally [0,5]', 'prop_1', 'finally [0,30]', 'globally [0,5]', 'prop_2', 'finally [0,30]', 'globally [0,5]', 'prop_3']
#
# natural language: At sometime between 1 to 25 stay at prop_1 for five seconds and always stay away from the prop_2 and prop_3 during 0 to 30 seconds.
# STL:  ['and', 'finally [1,25]', 'globally [0,5]', 'prop_1', 'globally [0,30]', 'and', 'negation', 'prop_2', 'negation', 'prop_3']
#
# natural language: Always (prop_1) if not (prop_2).
# STL:  ['imply', 'negation', 'prop_2', 'globally', 'prop_1']
#
# natural language: Always (prop_1) only if not (prop_2).
# STL:  ['imply', 'globally', 'prop_1', 'negation', 'prop_2']
#
# natural language: Avoid (prop_1) until (prop_2) lasts for 20 steps.
# STL:  ['until', 'negation', 'prop_1', 'globally [0, 20]', 'prop_2']
#
# natural language: Wait till (prop_1) is complete, then, (prop_2) for next 20 units.
# STL:  ['imply', 'finally', 'prop_1', 'globally [0, 20]', 'prop_2']
#
# natural language: ( prop_1 ) won't happen only if ( prop_2 )
# STL:  ['imply', 'negation', 'prop_1', 'prop_2']
#
# natural language: only under the case of ( prop_2 ), will ( prop_1 ) not happen
# STL:  ['imply', 'negation', 'prop_1', 'prop_2']
#
# natural language: ( prop_1 ) always follows with ( prop_2 ).
# STL:  ['globally', 'imply', 'prop_1', 'finally', 'prop_2']
#
# natural language: Maintain ( prop_1 ) until ( prop_2 ) is satisfied.
# STL:  ['globally', 'until', 'prop_1', 'prop_2']
#
# natural language: Go to ( prop_1 ) and always avoid both ( prop_2 ) and ( prop_3 ).
# STL:  ['and', 'finally', 'prop_1', 'globally', 'and', 'negation', 'prop_2', 'negation', 'prop_3']
#
# natural language: If ( prop_1 ) happens before ( prop_3 ) then start ( prop_2 ) and cancel ( prop_4 ) anytime within 0 to 10 timesteps.
# STL:  ['globally', 'imply', 'and', 'prop_1', 'finally', 'prop_3', 'and', 'prop_2', 'finally', 'negation', 'prop_4']
#
# natural language: For time steps between 0 and 20, until (prop_1) and (prop_2) is true, don’t start ( prop_3 ).
# STL:  ['globally [0,20]', 'until', 'negation', 'prop_3', 'and', 'prop_1', 'prop_2']
#
# natural language: If prop_1 and prop_2 and not prop_3 or prop_4, then prop_5 happens after 10 timesteps.
# STL:  ['imply', 'and', 'and', 'prop_1', 'prop_2', 'negation', 'or', 'prop_3', 'prop_4', 'finally [10,infinite]', 'prop_5']
#
# natural language: prop_1 for the first 5 timesteps then prop_2 thereafter.
# STL:  ['and', 'globally [0,5]', 'prop_1', 'globally [5,infinite]', 'prop_2']
#
# natural language: prop_1 and prop_2 and prop_3 and prop_4 and prop_5
# STL:  ['and', 'and', 'and', 'and', 'prop_1', 'prop_2', 'prop_3', 'prop_4', 'prop_5']
#
# natural language: prop_1, then prop_2 and stay there for 5 seconds, remember always prop_3.
# STL: ['and',  'finally', 'prop_1', 'and', 'finally', 'globally [0,5]', 'prop_2', 'globally', 'prop_3']
#
# natural language:"""
#
#
#
# user_prompt_2 = 'Based on your predicted STL ' + str(
#                         output_stl_1) + ' , the state sequence [[location, time]] of the generated trajectory is: ' + user_prompt_2_original + \
#                         '\n \nPlease print the initial instruction again and check whether this state sequence follows the instruction. ' \
#                         'Let us do it step by step, first specifically explain the semantic meanings of the instruction, and then list all the available rooms in the given environment, ' \
#                         'then determine the rooms planned to visit or avoid and whether the trajectory is consistent. ' \
#                         'Next modify or keep the final STL based on above analysis. First output your thinking steps and in the last line output the full final STL beginning with STL: . ' \
#                         '\nOutput:'
