#!/usr/bin/python3

from functools import reduce
import galois as gl
from constants import C, S, M, P
from typing import List

# 254-bit prime field of interest
#
# as it is here https://github.com/iden3/go-iden3-crypto/blob/e10db811aaa2570474548aef09f4468ed594e449/ff/element.go#L119
#
# I've noticed at module initialisation, execution of this line takes quite some time !
ff_p254 = gl.GF(
    21888242871839275222246405745257275088548364400416034343698204186575808495617)

N_ROUNDS_F = 8
N_ROUNDS_P = [56, 57, 56, 60, 60, 63, 64, 63, 60, 66, 60, 65, 70, 60, 64, 68]


def exp5_state(state):
    '''
        Raises each element of state to 5-th power
    '''
    return list(map(lambda e: e ** 5, state))


def ark(state, cnst):
    '''
        Updates state by adding round keys
    '''
    return list(map(lambda e: e[1] + ff_p254(cnst[e[0]]), enumerate(state)))


def mix(state, mat):
    '''
        Performs matrix - vector multiplication in following way,
        and updates state

        M = [
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3]
        ]

        V = [0, 1, 2, 3]

        Result = [[M]] * V = [
            sum([0, 0, 0, 0] * [0, 1, 2, 3]),
            sum([1, 1, 1, 1] * [0, 1, 2, 3]),
            sum([2, 2, 2, 2] * [0, 1, 2, 3]),
            sum([3, 3, 3, 3] * [0, 1, 2, 3]),
        ]
    '''
    return list(map(lambda e: reduce(lambda acc, cur: acc +
                                     (ff_p254(mat[cur[0]][e[0]]) * cur[1]), enumerate(state), ff_p254(0)), enumerate(state)))


def poseidon(input: List[int]) -> int:
    '''
        Poseidon Hash function implementation, works
        on aforementioned prime field, with input of size max 16

        Output hash is a 254-bit prime field element, which is returned
        as `int` (thanks to arbitrary precision arithmetic ability of Python !)

        This function is an adapted implementation of https://github.com/iden3/go-iden3-crypto/blob/f597e20569647d70f5eb4d3afde13c69d944a262/poseidon/poseidon.go#L58-L120
        and also takes some inspiration from https://github.com/iden3/circomlibjs/blob/fe43791f413959cbfcb485b8d27bafac5b9a473d/src/poseidon.js#L47-L101
    '''
    assert len(input) > 0, 'can\'t hash empty input'
    assert len(input) <= len(N_ROUNDS_P), f'max input length {len(N_ROUNDS_P)}'

    t = len(input) + 1  # = state size
    nRoundsF = N_ROUNDS_F
    nRoundsP = N_ROUNDS_P[t-2]

    # constants to be used
    C_ = C[t-2]
    S_ = S[t-2]
    M_ = M[t-2]
    P_ = P[t-2]

    # initialising state with input, while converting them to field element
    # no special check for whether input elements are within finite field or not, is performed, but I'd like to add one
    state = [ff_p254(0)] * t
    state[1:] = [ff_p254(i) for i in input[:]]

    state = ark(state, C_)
    for i in range((nRoundsF // 2) - 1):
        state = exp5_state(state)
        state = ark(state, C_[(i+1) * t:])
        state = mix(state, M_)

    state = exp5_state(state)
    state = ark(state, C_[(nRoundsF // 2) * t:])
    state = mix(state, P_)

    for i in range(nRoundsP):
        state[0] **= 5
        state[0] += ff_p254(C_[((nRoundsF // 2) + 1) * t + i])

        accState = ff_p254(0)
        for j in range(len(state)):
            accState += (ff_p254(S_[(2*t - 1) * i + j]) * state[j])

        for k in range(1, t):
            state[k] = state[k] + state[0] * \
                ff_p254(S_[(2*t - 1) * i + t + k - 1])

        state[0] = accState

    for i in range((nRoundsF // 2) - 1):
        state = exp5_state(state)
        state = ark(state, C_[((nRoundsF // 2 + 1) * t + nRoundsP + i * t):])
        state = mix(state, M_)

    state = exp5_state(state)
    state = mix(state, M_)

    # 254-bit hash output
    return int(state[0])


if __name__ == '__main__':
    print('nothing to execute here !')
