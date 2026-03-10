def sequential_try_with_nested_if(flag):
    prefix = 'keep'
    try:
        if flag:
            first = int('1')
        else:
            first = int('2')
    except Exception:
        first = -1
    try:
        second = 10 // first
    except Exception:
        second = 0
    return (prefix, first, second)
