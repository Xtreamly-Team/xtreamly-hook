// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {BaseHook} from "v4-periphery/src/utils/BaseHook.sol";
import {IPoolManager} from "v4-core/interfaces/IPoolManager.sol";
import {Hooks} from "v4-core/libraries/Hooks.sol";
import {PoolKey} from "v4-core/types/PoolKey.sol";
import {BalanceDelta} from "v4-core/types/BalanceDelta.sol";
import {Currency} from "v4-core/types/Currency.sol";
import {IxtrLogic} from "./IxtrLogic.sol";

/**
 * @title xtrHook
 * @notice Uniswap V4 hook that awards points for liquidity operations and swaps
 * @dev Uses a separate xtrLogic contract to avoid stack-too-deep errors
 */
contract xtrHook is BaseHook {
    IxtrLogic public xtrLogic;
    
    constructor(
        IPoolManager _poolManager,
        address _xtrLogic
    ) BaseHook(_poolManager) {
        xtrLogic = IxtrLogic(_xtrLogic);
    }
    
    /**
     * @notice Set a new xtrLogic contract
     * @param _xtrLogic Address of the new xtrLogic contract
     */
    function setXtrLogic(address _xtrLogic) external {
        // In a real contract, add access control here (onlyOwner)
        xtrLogic = IxtrLogic(_xtrLogic);
    }

    function getHookPermissions()
        public
        pure
        override
        returns (Hooks.Permissions memory)
    {
        return
            Hooks.Permissions({
                beforeInitialize: false,
                afterInitialize: false,
                beforeAddLiquidity: false,
                afterAddLiquidity: true,
                beforeRemoveLiquidity: false,
                afterRemoveLiquidity: true,
                beforeSwap: false,
                afterSwap: true,
                beforeDonate: false,
                afterDonate: false,
                beforeSwapReturnDelta: false,
                afterSwapReturnDelta: false,
                afterAddLiquidityReturnDelta: false,
                afterRemoveLiquidityReturnDelta: false
            });
    }

    function _afterAddLiquidity(
        address sender,
        PoolKey calldata key,
        IPoolManager.ModifyLiquidityParams calldata,
        BalanceDelta delta,
        BalanceDelta,
        bytes calldata
    ) internal override returns (bytes4, BalanceDelta) {
        // Delegate all logic to the xtrLogic contract
        xtrLogic.processLiquidityAdded(
            sender,
            delta,
            Currency.unwrap(key.currency0),
            Currency.unwrap(key.currency1)
        );
        
        return (BaseHook.afterAddLiquidity.selector, BalanceDelta.wrap(0));
    }
    
    function _afterRemoveLiquidity(
        address sender,
        PoolKey calldata key,
        IPoolManager.ModifyLiquidityParams calldata,
        BalanceDelta delta,
        BalanceDelta,
        bytes calldata
    ) internal override returns (bytes4, BalanceDelta) {
        // Delegate all logic to the xtrLogic contract
        xtrLogic.processLiquidityRemoved(
            sender,
            delta,
            Currency.unwrap(key.currency0),
            Currency.unwrap(key.currency1)
        );
        
        return (BaseHook.afterRemoveLiquidity.selector, BalanceDelta.wrap(0));
    }

    function _afterSwap(
        address sender,
        PoolKey calldata key,
        IPoolManager.SwapParams calldata,
        BalanceDelta delta,
        bytes calldata
    ) internal override returns (bytes4, int128) {
        // Delegate all logic to the xtrLogic contract
        xtrLogic.processSwap(
            sender,
            delta,
            Currency.unwrap(key.currency0),
            Currency.unwrap(key.currency1)
        );
        
        return (BaseHook.afterSwap.selector, 0);
    }
} 