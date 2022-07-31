def adjust_count_with_size_encoding(obj):
    encoding = obj['encoding']
    if len(encoding) >= 3 and 'size' in encoding:
        x_enc = encoding['x']
        y_enc = encoding['y']
        x_count = 'aggregate' in x_enc and x_enc['aggregate'] == 'count'
        y_count = 'aggregate' in y_enc and y_enc['aggregate'] == 'count'
        if bool(x_count) != bool(y_count):
            temp = encoding['size']
            if x_count:
                encoding['size'] = encoding['x']
                encoding['x'] = temp
            if y_count:
                encoding['size'] = encoding['y']
                encoding['y'] = temp


