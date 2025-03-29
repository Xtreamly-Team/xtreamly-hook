// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {BalanceDelta} from "v4-core/types/BalanceDelta.sol";

/**
 * @title IxtrLogic
 * @notice Interface for the xtrLogic contract
 */
interface IxtrLogic {
    function processLiquidityAdded(
        address user,
        BalanceDelta delta,
        address token0,
        address token1
    ) external;
    
    function processLiquidityRemoved(
        address user,
        BalanceDelta delta,
        address token0,
        address token1
    ) external;
    
    function processSwap(
        address user,
        BalanceDelta delta,
        address token0,
        address token1
    ) external;
} 