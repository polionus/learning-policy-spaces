import numpy as np



def get_write_value(y, i):
    return y[i]

### TODO: this is a horrible function, come and fix it later.
def step_function(x, y, max_x):
    y_p = np.array([None]*max_x)
    y_p[x] = y

    current_input_index = 0
    current_value = y_p[current_input_index]
    for index, value in enumerate(y_p):
        
        try:
            if y_p[index + 1] is not None:
            # current_input_index += 1
                current_value = y_p[index + 1]
        except:
            pass

        y_p[index] = current_value

    return y_p

if __name__ == "__main__":
    x = np.array([0, 12, 15, 20])
    y = np.array([-1, 0, 13, 17])
    max_x = 25

    print(step_function(x, y, max_x=max_x))

# def step_function(x, y, max_value):
#     assert max_value>0
#     x_values = np.arange(0, max_value)
#     return y[np.searchsorted(x, x_values, side='right')]


