import re
import ast
import itertools


def parse_command_with_lists(cmd: str):
    """
    Parse command string and extract parameters with list values.
    Returns base command and dictionary of parameter variations.
    Boolean flags are handled as flags without values.
    """
    # Extract base command (everything before first --)
    base_cmd = re.split(r'\s+--', cmd)[0]
    
    # Split into tokens
    tokens = cmd[len(base_cmd):].strip().split()
    
    param_variations = {}
    boolean_flags = []
    i = 0
    
    while i < len(tokens):
        if tokens[i].startswith('--'):
            param_name = tokens[i][2:]  # Remove '--'
            
            # Check if next token exists and is not another flag
            if i + 1 < len(tokens) and not tokens[i + 1].startswith('--'):
                param_value = tokens[i + 1]
                
                if param_value.startswith('['):
                    # Handle list values (may span multiple tokens)
                    list_str = param_value
                    j = i + 1
                    while not list_str.endswith(']') and j < len(tokens) - 1:
                        j += 1
                        list_str += ' ' + tokens[j]
                    
                    try:
                        values = ast.literal_eval(list_str)
                        if not isinstance(values, list):
                            values = [values]
                        param_variations[param_name] = values
                    except:
                        param_variations[param_name] = [list_str]
                    
                    i = j + 1
                else:
                    # Single value
                    param_variations[param_name] = [param_value]
                    i += 2
            else:
                # Boolean flag (no value after it)
                boolean_flags.append(param_name)
                i += 1
        else:
            i += 1
    
    return base_cmd, param_variations, boolean_flags

def generate_command_combinations(base_cmd: str, param_variations: dict, boolean_flags: list):
    """
    Generate all combinations of parameters.
    Boolean flags are added to all commands.
    """
    if not param_variations:
        # Only boolean flags, no variations
        cmd_parts = [base_cmd]
        for flag in boolean_flags:
            cmd_parts.append(f"--{flag}")
        return [" ".join(cmd_parts)]
    
    param_names = list(param_variations.keys())
    param_values = [param_variations[name] for name in param_names]
    
    commands = []
    for combination in itertools.product(*param_values):
        cmd_parts = [base_cmd]
        
        # Add boolean flags first
        for flag in boolean_flags:
            cmd_parts.append(f"--{flag}")
        
        # Add parameters with values
        for param_name, param_value in zip(param_names, combination):
            cmd_parts.append(f"--{param_name} {param_value}")
        
        commands.append(" ".join(cmd_parts))
    
    return commands


def get_all_commands(cmd: str):
    base_cmd, param_variations, boolean_flags = parse_command_with_lists(cmd)
    return generate_command_combinations(base_cmd, param_variations, boolean_flags)