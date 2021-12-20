from typing import Dict, List
import re

def get_word_frequencies(doc: str) -> Dict[str, int]:
    """Extracts word frequencies from a document.
    
    Args:
        doc: Document content given as a string.
    
    Returns:
        Dictionary with words as keys and their frequencies as values.
    """
    # TODO
    splitted = re.split(r'[\s.,:;?!]+',doc) #Regex for splitting on all the given delimiters
    if "" in splitted: splitted.remove("")
    outputDict = {}
    for words in splitted:
        outputDict[words] = splitted.count(words)
    return outputDict


def get_word_feature_vector(
    word_frequencies: Dict[str, int], vocabulary: List[str]
) -> List[int]:
    """Creates a feature vector for a document, comprising word frequencies 
        over a vocabulary.
    
    Args:
        word_frequencies: Dictionary with words as keys and frequencies as 
            values.
        vocabulary: List of words.
    
    Returns:
        List of length `len(vocabulary)` with respective frequencies as values.
    """
    # TODO
    outputlist = []
    for words in vocabulary:
        freq = word_frequencies.get(words) if word_frequencies.get(words) else 0
        outputlist.append(freq)
    return outputlist
