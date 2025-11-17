from itertools import product


def CMD(script_name):
    return list(("python", f"src/{script_name}.py"))

def parse_config(config: dict, script_name: str): 
    list_args = {}
    static_args = {}
    cmds = []
    args_list = []
    cmd = CMD(script_name)

    ## I have myself a dictionary, and I would like to see which values are lists, and which values are not:
    for key, value in config.items():

        if isinstance(value,list):
            list_args[key] = value
        else:
            static_args[key] = value
            
    if not list_args:
        # cmd = ["python", f"src/{script_name}.py"]
        for k, v in static_args.items():

            if isinstance(v, bool):
                if v:
                    cmd += [f"--{k}"]
            else:
                cmd += [f"--{k}", str(v)]
        cmds.append(cmd)
        args_list.append(static_args)
        return cmds, args_list
        # subprocess.run(cmd)
    else:
        keys, values = zip(*list_args.items()) #This will let us create the correct cross-product
        for combination in product(*values):
            args = static_args.copy()
            
            #This is the new config, and we can get its meta data and check for has collision here.
            args.update(dict(zip(keys, combination))) ### What does this do?
            for k, v in args.items():
                if isinstance(v, bool):
                    if v:
                        cmd += [f"--{k}"]
                else:
                    cmd += [f"--{k}", str(v)]

            cmds.append(cmd)
            args_list.append(args)
            cmd = CMD(script_name)
            
            # subprocess.run(cmd)
        return cmds, args_list #By not running and returning the the commands, we could run them outside. 

