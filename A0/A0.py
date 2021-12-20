from typing import List, Union


def get_unique_elements(
    lst: List[Union[str, int]], n: int = 1
) -> List[Union[str, int]]:
    """Given a list of elements returns those that repeat at least n times. The
    output list should contain all unique elements and they should be returned
    in the same order as they first appear in the input list.

    Args:
        lst: Input list
        n (optional): Minimum number of times an element should be repeated to 
            be returned. Defaults to 1.

    Returns:
        List of unique items 
    """
    unique = list(set(lst))
    print(unique)
    if n == 1:
        return unique
    else:
        outputList = []
        for ele in unique:
            if( count_occ(lst, ele) >= n):
                outputList.append(ele)
        
        return outputList


#Helper function that counts the occurences of one element in a list and returns this as
#an integer
def count_occ(lst:List[Union[str, int]], item) -> int:
    x = 0
    for e in lst:
        if e == item:
            x += 1
    return x 