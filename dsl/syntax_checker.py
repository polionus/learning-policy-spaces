import jax 
import jax.numpy as jnp
import equinox as eqx
from typing import Tuple
from dataclasses import replace

### TODO: I basically have to read this, comment on it and implement changes so that it works with jax


### some global constants. These are the states of the syntax checker as a push-down automata, to check the validity of the syntax of the context-free grammar.
STATE_MANDATORY_NEXT = 0
STATE_ACT_NEXT = 1
STATE_CSTE_NEXT = 2
STATE_BOOL_NEXT = 3
STATE_POSTCOND_OPEN_PAREN = 4



### REQUIRED changes: 1. get rid of dynamic python datastructures, like sets, and make the inclusion checks work with jax.
### UPDATE: there is a function called jnp.isin() that allows jitting! 

### UPDATE: You need to set the length of the largest program as an input, to make a fixed size array.
### We want to create a look-up table that takes in the index of the check (using the integer index map), 
### and then look up the transformations necessary to the checker state. 


### So we will have: 

##1. An LUT
##2. A checker state




### TODO: I need to verify that all the assert statements are still valid. (keep the assert statements)
### UPDATE: I am now understanding that I might not be able to do this, and I should first check the 
### Syntax of the input string directly before passing them.

### TODO: Change the replace function with eqx.tree_at?


### They are converted to sets, and used to check for inclusion.
open_paren_token = ["m(", "c(", "r(", "w(", "i(", "e("]
close_paren_token = ["m)", "c)", "r)", "w)", "i)", "e)"]

### used to construct a set, and a union with other sets, an inclusion check. 
flow_leads = ["REPEAT", "WHILE", "IF", "IFELSE"]

### used to construct a set, and an inclusion check
flow_need_bool = ["WHILE", "IF", "IFELSE"]

### used to construct a set, and inclusion checks
acts = ["move", "turnLeft", "turnRight", "pickMarker", "putMarker"]

### Used to construct a set, and inclusion checks
bool_check = [
    "markersPresent",
    "noMarkersPresent",
    "leftIsClear",
    "rightIsClear",
    "frontIsClear",
]

### Never Used in the logic here.
next_is_act = ["i(", "e(", "r(", "m(", "w("]

### Used to construct a set, inlusion checks, and to further construct a dicitonary
postcond_open_paren = ["i(", "w("]

### Used in inclusion checks.
possible_mandatories = ["DEF", "run", "c)", "ELSE", "<pad>"] + open_paren_token

### Never used.
all_tokens = [
    "DEF",
    "run",
    "m(",
    "m)",
    "move",
    "turnRight",
    "turnLeft",
    "pickMarker",
    "putMarker",
    "r(",
    "r)",
    "R=0",
    "R=1",
    "R=2",
    "R=3",
    "R=4",
    "R=5",
    "R=6",
    "R=7",
    "R=8",
    "R=9",
    "R=10",
    "R=11",
    "R=12",
    "R=13",
    "R=14",
    "R=15",
    "R=16",
    "R=17",
    "R=18",
    "R=19",
    "REPEAT",
    "c(",
    "c)",
    "i(",
    "i)",
    "e(",
    "e)",
    "IF",
    "IFELSE",
    "ELSE",
    "frontIsClear",
    "leftIsClear",
    "rightIsClear",
    "markersPresent",
    "noMarkersPresent",
    "not",
    "w(",
    "w)",
    "WHILE",
]

### Standardize how to initialize the stacks
class InitStack:

    i_need_else_stack: jax.Array = jnp.array(128 * [False], dtype = bool)
    to_close_stack: jax.Array = jnp.array(128 * [-1])


### Keep the state object just a normal class with no methods.
### USE eqx.Module to register the state as a PyTree
class CheckerState(eqx.Module):
    
    state: int 
    next_mandatory: int 
    i_need_else_stack_pos: int 
    to_close_stack_pos: int 
    c_deep: int 
    next_actblock_open: int
    i_need_else_stack: jax.Array
    to_close_stack: jax.Array



    def __init__(self,
                 state: int, 
                 next_mandatory: int, 
                 i_need_else_stack_pos: int, 
                 to_close_stack_pos: int, 
                 c_deep: int, 
                 next_actblock_open: int,
                 i_need_else_stack: jax.Array,
                 to_close_stack: jax.Array
                 ):
        self.state = state
        self.next_mandatory = next_mandatory
        self.i_need_else_stack_pos = i_need_else_stack_pos
        self.to_close_stack_pos = to_close_stack_pos
        self.c_deep = c_deep
        self.next_actblock_open = next_actblock_open
        self.i_need_else_stack =  i_need_else_stack
        self.to_close_stack = to_close_stack

    
def set_i_need_else_stack(checker_state:CheckerState, x: jax.Array):
    return replace(checker_state, i_need_else_stack = x)

    
def set_to_close_stack(checker_state: CheckerState, x: jax.Array):
    return replace(checker_state, to_close_stack = x)


def push_closeparen_to_stack(checker_state: CheckerState, close_paren: int):

    return replace(checker_state, to_close_stack_pos = checker_state.to_close_stack_pos + 1, 
                   to_close_stack = checker_state.to_close_stack.at[checker_state.to_close_stack_pos].set(close_paren))



def pop_close_paren(checker_state: CheckerState) -> Tuple[int, CheckerState]:
        to_ret = checker_state.to_close_stack[checker_state.to_close_stack_pos]
        
        checker_state = replace(checker_state, to_close_stack_pos = checker_state.to_close_stack_pos - 1)
        return to_ret, checker_state



def paren_to_close(checker_state: CheckerState):
    return checker_state.to_close_stack[checker_state.to_close_stack_pos]


def make_next_mandatory(checker_state: CheckerState, next_mandatory: int):
    return replace(checker_state, state = STATE_MANDATORY_NEXT, next_mandatory = next_mandatory)


def make_bool_next(checker_state: CheckerState):

    return replace(checker_state, state = STATE_BOOL_NEXT, c_deep = checker_state.c_deep + 1)

    # return CheckerState(STATE_BOOL_NEXT, 
    #                     checker_state.next_mandatory,
    #                     checker_state.i_need_else_stack_pos,
    #                     checker_state.to_close_stack_pos,
    #                     checker_state.c_deep + 1, 
    #                     checker_state.next_actblock_open,
    #                     i_need_else_stack=checker_state.i_need_else_stack,
    #                     to_close_stack=checker_state.to_close_stack
    #                     )


def make_act_next(checker_state: CheckerState):    
    return replace(checker_state, state = STATE_ACT_NEXT)






def close_cond_paren(checker_state:CheckerState):
    _state = jax.lax.cond(checker_state.c_deep == 0, lambda: STATE_POSTCOND_OPEN_PAREN, lambda: STATE_MANDATORY_NEXT)
    return replace(checker_state, state = _state, c_deep = checker_state.c_deep - 1)



def push_needelse_stack(checker_state: CheckerState, need_else: bool):

    return replace(checker_state,i_need_else_stack = checker_state.i_need_else_stack.at[checker_state.i_need_else_stack_pos].set(need_else), 
                   i_need_else_stack_pos = checker_state.i_need_else_stack_pos + 1)


def pop_needelse_stack(checker_state: CheckerState) -> Tuple[bool, CheckerState]:
    to_ret = checker_state.i_need_else_stack[checker_state.i_need_else_stack_pos]
    checker_state = replace(checker_state, i_need_else_stack_pos = checker_state.i_need_else_stack_pos -1)

    return to_ret, checker_state
                        

def set_next_actblock(checker_state: CheckerState, next_actblock: int):
    return replace(checker_state, next_actblock_open = next_actblock)
    
def make_next_cste(checker_state: CheckerState):
    return replace(checker_state, state = STATE_CSTE_NEXT)
    

#### Forward check functions:
def is_vocab_c_open_tkn(cond: bool, checker_state: CheckerState ):
    return jax.lax.cond(cond, make_bool_next, make_act_next, checker_state)


### This is only for readability reasons, it doesn't need to exist.
class SyntaxVocabulary(object):

    def __init__(
        self,
        def_tkn,
        run_tkn,
        m_open_tkn,
        m_close_tkn,
        else_tkn,
        e_open_tkn,
        c_open_tkn,
        c_close_tkn,
        i_open_tkn,
        i_close_tkn,
        while_tkn,
        w_open_tkn,
        repeat_tkn,
        r_open_tkn,
        not_tkn,
        pad_tkn,
    ):
        self.def_tkn = def_tkn
        self.run_tkn = run_tkn
        self.m_open_tkn = m_open_tkn
        self.m_close_tkn = m_close_tkn
        self.else_tkn = else_tkn
        self.e_open_tkn = e_open_tkn
        self.c_open_tkn = c_open_tkn
        self.c_close_tkn = c_close_tkn
        self.i_open_tkn = i_open_tkn
        self.i_close_tkn = i_close_tkn
        self.while_tkn = while_tkn
        self.w_open_tkn = w_open_tkn
        self.repeat_tkn = repeat_tkn
        self.r_open_tkn = r_open_tkn
        self.not_tkn = not_tkn
        self.pad_tkn = pad_tkn

class PySyntaxChecker:

    def __init__(self, T2I: dict, only_structure: bool = False):
        # check_type(args.no_cuda, bool)
        self.vocab = SyntaxVocabulary(
            T2I["DEF"],
            T2I["run"],
            T2I["m("],
            T2I["m)"],
            T2I["ELSE"],
            T2I["e("],
            T2I["c("],
            T2I["c)"],
            T2I["i("],
            T2I["i)"],
            T2I["WHILE"],
            T2I["w("],
            T2I["REPEAT"],
            T2I["r("],
            T2I["not"],
            T2I["<pad>"],
        )
        ### So probably, I can change the sets to jnp arrays, and then change the inclusion checks to lax.switches? I can then not touch the for loops here, since they are only
        ### run once, they don't (and won't) need to be jitted. 
        self.vocab_size = len(T2I)
        
        self.only_structure = only_structure
        self.open_parens = jnp.array([T2I[op] for op in open_paren_token])
        
         # only an inclusion check. And this set set is a set of indices
        self.close_parens = jnp.array([T2I[op] for op in close_paren_token]) # these sets are all indices, so I can simply make them jnp.arrays and use isin. it should be just as fast, or faster because of jit
        self.if_statements = jnp.array([T2I[tkn] for tkn in ["IF", "IFELSE"]])
        #### Let's turn these dictionaries into arrays.



        #self.op2cl = {} ### TODO: Make sure this works.
        self.op2cl = jnp.full(shape= (self.vocab_size), fill_value=-1)
        for op, cl in zip(open_paren_token, close_paren_token):
            self.op2cl = self.op2cl.at[T2I[op]].set(T2I[cl])

        # self.need_else = {T2I["IF"]: False, T2I["IFELSE"]: True} ### This is a PyTree!
        # self.need_else
        self.if_idx = T2I["IF"]
        self.ifelse_idx = T2I['IFELSE']
        self.flow_lead = jnp.array([T2I[flow_lead_tkn] for flow_lead_tkn in flow_leads])

        ### Once appended to in the init, but only inclusion checks in the main traning

        ### The effects add gets something added to it, but the other ones either stay empty, or on

        if self.only_structure:
            self.effect_acts = []
            self.range_cste = jnp.array()
            self.bool_checks = jnp.array()
        else:
            self.effect_acts = jnp.array([T2I[act_tkn] for act_tkn in acts])
            self.range_cste = jnp.array(
                [idx for tkn, idx in T2I.items() if tkn.startswith("R=")]
            )
            self.bool_checks = jnp.array([T2I[bcheck] for bcheck in bool_check])
        
        ###TODO: make this part jax compatible
        ##ATTENTION: inclusion check
        if "<HOLE>" in T2I.keys():
            ### a dynamic set that would get a new element
            self.effect_acts.append(T2I["<HOLE>"])

        self.effect_acts = jnp.array(self.effect_acts)

        ## Union operation between sets, only used in assert statement
        #self.act_acceptable = self.effect_acts | self.flow_lead | self.close_parens
        self.flow_needs_bool = jnp.array([T2I[flow_tkn] for flow_tkn in flow_need_bool])
        self.postcond_open_paren = jnp.array([T2I[op] for op in postcond_open_paren])

        # tt = torch.cuda if "cuda" in self.device.type else torch
        

        ### This can also be changed to an array, since the keys are simply integers, which can be used as indices of the array.
        ### the best way to predetermine the size of the array is search go through the values of t2i and get the maximum. This will be the maximum index.
        
        self.mandatories_mask = jnp.full(shape = (len(possible_mandatories), 1, 1, self.vocab_size), fill_value=True, dtype = jnp.bool_)
        for mand_tkn in possible_mandatories:
            mask = jnp.ones((1,1,self.vocab_size), dtype=jnp.bool) ### This is all trues, except when 
            mask = mask.at[0, 0, T2I[mand_tkn]].set(False)
            self.mandatories_mask = self.mandatories_mask.at[T2I[mand_tkn]].set(mask)
     
        ## TODO: I have to rename the masks and their locations.
                # self.act_next_masks = {}
        self.act_next_masks = jnp.full(shape=(len(self.close_parens), 1, 1, self.vocab_size), fill_value=True, dtype = jnp.bool_)
        for close_tkn in self.close_parens:
            # mask = tt.BoolTensor(1, 1, self.vocab_size).fill_(1)
            mask = jnp.ones((1,1,self.vocab_size), dtype=jnp.bool)
            mask = mask.at[0, 0, close_tkn].set(False)
            for effect_idx in self.effect_acts:
                mask = mask.at[0, 0, effect_idx].set(False)
            for flowlead_idx in self.flow_lead:
                mask = mask.at[0, 0, flowlead_idx].set(False)
            self.act_next_masks = self.act_next_masks.at[int(close_tkn)].set(mask)

        ### Make range mask (works, perhaps better to use boolean indicators) ## this will just work fine.
        self.range_mask = jnp.ones((1,1,self.vocab_size), dtype = jnp.bool_)
        for ridx in self.range_cste:
            self.range_mask = self.range_mask.at[0, 0, ridx].set(False)
        
        
        ### Make boolnext mask (works)
        self.boolnext_mask = jnp.ones((1,1,self.vocab_size), dtype = jnp.bool_)
        for bcheck_idx in self.bool_checks:
            self.boolnext_mask = self.boolnext_mask.at[0, 0, bcheck_idx].set(False)
        self.boolnext_mask = self.boolnext_mask.at[0, 0, self.vocab.not_tkn].set(False)
        
        
        ### Make Post Cond masks
        # self.postcond_open_paren_masks = {}
        self.postcond_open_paren_masks = jnp.full(shape = (len(self.postcond_open_paren), 1, 1, self.vocab_size), fill_value=True, dtype=jnp.bool_)
        for tkn in self.postcond_open_paren:
            mask = jnp.ones((1,1,self.vocab_size), dtype = jnp.bool)
            mask = mask.at[0, 0, tkn].set(False)
            self.postcond_open_paren_masks.at[int(tkn)].set(mask)

    def need_else(self, tkn_idx: int) -> bool:
    # self.need_else = {T2I["IF"]: False, T2I["IFELSE"]: True} ### This is a PyTree!
        conds = jnp.array([
            tkn_idx == self.if_idx,
            tkn_idx == self.ifelse_idx
        ])

        case = jnp.select(conds, jnp.arange(len(conds)))

        def _on_if(): return False
        def _on_ifelse(): return True
        
        branches = (
            _on_if,
            _on_ifelse,
        )


        return jax.lax.switch(case, branches)


    def forward(self, state: CheckerState, new_idx: int):
        # Whenever we have a problem in having too many inputs, we should define the input in the local context of the function, and then define an inner function that doesn't get the 
        # input
        checker_state = state

        ### So this format of defining functions inside the context of other functions allows me to rewrite multiple 
        ### steps and layers of conditionals, and also adapt their input arguments to only get the state.
        def _push_if_open(checker_state: CheckerState) -> CheckerState:
            is_open = jnp.isin(new_idx, self.open_parens, assume_unique=True)
           
            def _push(checker_state: CheckerState) -> CheckerState: 
                close_tok = self.op2cl[new_idx]
                return push_closeparen_to_stack(checker_state, close_tok)
            
            def _noop(checker_state: CheckerState) -> CheckerState: return checker_state
            
            return jax.lax.cond(is_open, _push, _noop, checker_state)

        def _pop_if_close(checker_state: CheckerState) -> CheckerState: 
            is_close = jnp.isin(new_idx, self.close_parens)

            def _pop(checker_state: CheckerState) -> CheckerState:
                tok, checker_state = pop_close_paren(checker_state)
                return checker_state, tok
        
            def _noop(checker_state: CheckerState) -> CheckerState:
                return checker_state, jnp.int32(-1)
            
             
            return jax.lax.cond(is_close, _pop, _noop, checker_state)
        
        checker_state = _push_if_open(checker_state)
        checker_state, tok = _pop_if_close(checker_state)


    
        conds = jnp.array([
            (checker_state.state == STATE_MANDATORY_NEXT),
            (checker_state.state == STATE_ACT_NEXT),
            (checker_state.state == STATE_CSTE_NEXT),
            (checker_state.state == STATE_BOOL_NEXT),
            (checker_state.state == STATE_POSTCOND_OPEN_PAREN)
        ])

        case = jnp.select(conds, jnp.arange(len(conds)), default=len(conds))

        def _state_mandatory(checker_state: CheckerState) -> CheckerState: 

            conds = jnp.array([
                (new_idx == self.vocab.def_tkn),
                (new_idx == self.vocab.run_tkn),
                (new_idx == self.vocab.else_tkn),
                (jnp.isin(new_idx,self.open_parens)),
                (new_idx == self.vocab.c_close_tkn),
                (new_idx == self.vocab.pad_tkn)
            ])

            case = jnp.select(conds, jnp.arange(len(conds)), default=len(conds)) ### the default is the last else statement

            def _on_def(checker_state: CheckerState) -> CheckerState: return make_next_mandatory(checker_state, self.vocab.run_tkn)
            def _on_run(checker_state: CheckerState) -> CheckerState: return make_next_mandatory(checker_state, self.vocab.m_open_tkn)
            def _on_else(checker_state: CheckerState) -> CheckerState: return make_next_mandatory(checker_state, self.vocab.e_open_tkn)
            def _on_open(checker_state: CheckerState) -> CheckerState: return jax.lax.cond(new_idx == self.vocab.c_open_tkn, make_bool_next, make_act_next, checker_state)
            def _on_close(checker_state: CheckerState) -> CheckerState: return close_cond_paren(checker_state)
            def _on_default(checker_state: CheckerState) -> CheckerState: return checker_state

            branches = (
                        _on_def,
                        _on_run,
                        _on_else,
                        _on_open,
                        _on_close, 
                        _on_default 
            )

            return jax.lax.switch(case, branches, checker_state)
        
            
        def _state_act_next(checker_state: CheckerState) -> CheckerState: 

            conds = jnp.array([
                (jnp.isin(new_idx, self.flow_needs_bool)),
                (new_idx == self.vocab.repeat_tkn),
                (jnp.isin(new_idx,self.effect_acts)),
                (jnp.isin(new_idx, self.close_parens)),
            ])            

            case = jnp.select(conds, jnp.arange(len(conds)), default=len(conds))

            def _on_flow_bool(checker_state: CheckerState) -> CheckerState: 

                checker_state = make_next_mandatory(checker_state, self.vocab.c_open_tkn)

                conds = jnp.array([
                    (jnp.isin(new_idx, self.if_statements)),
                    (new_idx == self.vocab.while_tkn),
                ])

                case = jnp.select(conds, jnp.arange(len(conds)), default = len(conds))

                def _on_if_statements(checker_state: CheckerState) -> CheckerState: 
                    checker_state = push_needelse_stack(checker_state, self.need_else(new_idx))
                    checker_state = set_next_actblock(checker_state, self.vocab.i_open_tkn)
                    return checker_state

                def _on_while(checker_state: CheckerState) -> CheckerState: return set_next_actblock(checker_state, self.vocab.w_open_tkn)
                def _on_default(checker_state: CheckerState) -> CheckerState: return checker_state

                branches = (
                    _on_if_statements,
                    _on_while,
                    _on_default
                )

                return jax.lax.switch(case, branches, checker_state)
            
            def _on_repeat(checker_state: CheckerState) -> CheckerState: return make_next_cste(checker_state)
            def _on_effect_acts(checker_state: CheckerState) -> CheckerState: return checker_state
            def _on_close(checker_state: CheckerState) -> CheckerState:

                conds = jnp.array([
                    (new_idx == self.vocab.i_close_tkn),
                    (new_idx == self.vocab.m_close_tkn),
                ])

                case = jnp.select(conds, jnp.arange(len(conds)), default=len(conds))

                def _on_i_close(checker_state: CheckerState) -> CheckerState:
                    need_else, checker_state = pop_needelse_stack(checker_state)

                    def _make_next_mandatory_else(checker_state: CheckerState) -> CheckerState:
                        return make_next_mandatory(checker_state, self.vocab.else_tkn)

                    return jax.lax.cond(need_else, _make_next_mandatory_else, make_act_next, checker_state)
                
                def _on_m_close(checker_state: CheckerState) -> CheckerState: return make_next_mandatory(checker_state, self.vocab.pad_tkn)
                    
                def _on_default(checker_state: CheckerState) -> CheckerState: return make_act_next(checker_state)

                branches = (_on_i_close,
                            _on_m_close,
                            _on_default)
                
                return jax.lax.switch(case, branches, checker_state)
            def _on_default(checker_state: CheckerState) -> CheckerState: return checker_state

            branches = (
                _on_flow_bool,
                _on_repeat,
                _on_effect_acts,
                _on_close,
                _on_default
            )

            return jax.lax.switch(case, branches, checker_state)
    
        def _state_cste_next(checker_state: CheckerState) -> CheckerState: return make_next_mandatory(checker_state, self.vocab.r_open_tkn)

        def _state_bool_next(checker_state: CheckerState) -> CheckerState:
            
            conds = jnp.array([
                (jnp.isin(new_idx, self.bool_checks)),
                (new_idx == self.vocab.not_tkn)
            ])

            case = jnp.select(conds, jnp.arange(len(conds)), default=len(conds))

            def _on_bool_checks(checker_state: CheckerState) -> CheckerState: return make_next_mandatory(checker_state, self.vocab.c_close_tkn)
            def _on_not(checker_state: CheckerState) -> CheckerState: return make_next_mandatory(checker_state, self.vocab.c_open_tkn)
            def _on_default(checker_state: CheckerState) -> CheckerState: return checker_state

            branches = (
                _on_bool_checks, 
                _on_not,
                _on_default
            )

            return jax.lax.switch(case, branches, checker_state)

        def _state_postcond_open_paren(checker_state: CheckerState) -> CheckerState: return make_act_next(checker_state)

        def _on_default(chekcer_state: CheckerState) -> CheckerState: return chekcer_state

        branches = (
            _state_mandatory,
            _state_act_next,
            _state_cste_next,
            _state_bool_next,
            _state_postcond_open_paren,
            _on_default
        )

        return jax.lax.switch(case, branches, checker_state)

    ### This can simply be replaced with lax.switch statement.  

    #### What does the allowed tokens return ?
    def allowed_tokens(self, checker_state: CheckerState) -> jax.Array:

        conds = jnp.array([
            checker_state.state == STATE_MANDATORY_NEXT,
            checker_state.state == STATE_ACT_NEXT,
            checker_state.state == STATE_CSTE_NEXT,
            checker_state.state == STATE_BOOL_NEXT,
            checker_state.state == STATE_POSTCOND_OPEN_PAREN
        ])

        case = jnp.select(conds, jnp.arange(len(conds)), default=len(conds))

        def _on_mandatory(checker_state: CheckerState) -> jax.Array : return self.mandatories_mask[checker_state.next_mandatory]
        def _on_act(checker_state: CheckerState) -> jax.Array: return self.act_next_masks[paren_to_close(checker_state)]
        def _on_cste(checker_state: CheckerState) -> jax.Array: return self.range_mask
        def _on_bool(checker_state: CheckerState) -> jax.Array: return self.boolnext_mask
        def _on_postcond(checker_state: CheckerState) -> jax.Array: return self.postcond_open_paren_masks[checker_state.next_actblock_open]

        branches = (
            _on_mandatory,
            _on_act,
            _on_cste,
            _on_bool,
            _on_postcond,
        )

        return jax.lax.switch(case, branches, checker_state)

    ### TODO: This is the for loop used to check the feasiblity of the tokens, and I should use jax constructs to implemenet it. 
    ### Update: the input sequence has been written in such a way that the sequence mask is only passed once.
    #@eqx.filter_jit
    def get_sequence_mask(self, checker_state: CheckerState, inp_sequence: list) -> Tuple[CheckerState, jax.Array]:
        inp_sequence = jnp.array(inp_sequence)
        checker_state = self.forward(checker_state, inp_sequence[0])
        return checker_state, jnp.squeeze(self.allowed_tokens(checker_state))

    def get_initial_checker_state(self):
        return CheckerState(STATE_MANDATORY_NEXT, self.vocab.def_tkn, -1, -1, 0, -1, InitStack.i_need_else_stack, InitStack.to_close_stack)

    def get_initial_checker_state2(self):
        return CheckerState(STATE_MANDATORY_NEXT, self.vocab.m_open_tkn, -1, -1, 0, -1, InitStack.i_need_else_stack, InitStack.to_close_stack)
