import json
import pdb

# example = '{\n    "CLUES" : ["Symantec", "Veritas "Software" Corp.", "buy", "\\$13.5 billion", "security software", "backup and recovery software", "expand", "New York",],\n    "REASONING" : "The input mentions the acquisition of Symantec by Veritas Software Corp. in the context of expanding their respective markets.",\n    "TOPIC" : "Business"\n}'

BAD_TOKENS = ['"', '\\']

PATTERN_0 = '{\n    "CLUES" : ' 
PATTERN_1 = ',\n    "REASONING" : ' 
PATTERN_2 = ',\n    "SENTIMENT" : ' 
PATTERN_3 = ',\n    "TOPIC" : ' 
PATTERN_4 = '\n}' 

def _check_validity(s):
    if not PATTERN_0 in s:
        return None
    if not PATTERN_1 in s:
        return None
    if not PATTERN_4 in s:
        return None    
    if PATTERN_2 in s:
        return PATTERN_2    
    if PATTERN_3 in s:
        return PATTERN_3    
    return None
    
def _extract_clues(s, mode):
    clues = s.split(PATTERN_0)[1].split(PATTERN_1)[0] 
    if clues[0] != '[':
        clues = '[' + clues
    
    if clues[-1] != ']':
        clues = clues + ']'
    
    return clues
    
def _extract_reasoning(s, mode):
    reasoning = s.split(PATTERN_1)[1].split(mode)[0]
    if reasoning[0] != '"':
        reasoning = '"' + reasoning
    
    if reasoning[-1] != '"':
        reasoning = reasoning + '"'

    return reasoning

def _extract_answer(s, mode):
    answer = s.split(mode)[1].split(PATTERN_4)[0]
    if answer[0] != '"':
        answer = '"' + answer
    
    if answer[-1] != '"':
        answer = answer + '"'

    return answer

def _modify_clues(clues):
    clues = clues[1:-1]
    if clues[-1] == ',':
        clues = clues[:-1]
    
    clues = clues.split('", "')
    for t in BAD_TOKENS:
        clues = [clue.replace(t, '') for clue in clues]
    
    return '["' + '", "'.join(clues) + '"]'
    

def _modify_reasoning(reasoning):
    reasoning = reasoning[1:-1]
    for t in BAD_TOKENS:
        reasoning = reasoning.replace(t, '')
    
    return '"' + reasoning + '"'
    

def reconstruct(s):
    mode = _check_validity(s)
    if not mode:
        return None
    
    clues = _extract_clues(s, mode)
    clues = _modify_clues(clues)
    
    reasoning = _extract_reasoning(s, mode)
    reasoning = _modify_reasoning(reasoning)
    
    answer = _extract_answer(s, mode)
    
    return  PATTERN_0 + clues + PATTERN_1 + reasoning + mode + answer + PATTERN_4


# print(example)
# print(reconstruct(example))