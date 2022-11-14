def twoNumberSum(array, targetSum):
    hmap = {}
    found_list = []
    for x in array:
        y = targetSum - x
        if y in hmap.keys():
            found_list.append([x, y])
        else:
            hmap[y] = True

    return found_list

array = [3, 5, -4, 8, 11, 1, -1, 6]
targetSum = 10

twoNumberSum(array, targetSum)
