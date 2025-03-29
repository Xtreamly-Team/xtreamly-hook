// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Ownable} from "@openzeppelin/contracts/access/Ownable.sol";
import {ERC20} from "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import {BalanceDelta} from "v4-core/types/BalanceDelta.sol";
import {Currency} from "v4-core/types/Currency.sol";

/**
 * @title xtrLogic
 * @notice Contract to handle the logic for points allocation and event emission
 * @dev This is separated from the hook to avoid stack-too-deep errors
 */
contract xtrLogic is Ownable, ERC20 {
    // Enhanced event with token amounts
    event LiquidityAdded(
        address indexed user,
        uint256 points,
        uint256 totalUserPoints,
        int256 token0Amount,
        int256 token1Amount,
        address token0,
        address token1,
        uint256 timestamp
    );
    
    event LiquidityRemoved(
        address indexed user,
        uint256 points,
        uint256 totalUserPoints,
        int256 token0Amount,
        int256 token1Amount,
        address token0,
        address token1,
        uint256 timestamp
    );
    
    event SwapExecuted(
        address indexed user,
        uint256 points,
        uint256 totalUserPoints,
        int256 token0Delta,
        int256 token1Delta,
        address token0,
        address token1,
        uint256 timestamp
    );

    mapping(address => uint256) public userPoints;
    uint256 public totalPoints;
    
    // Only the hook contract can call certain functions
    address public hookContract;
    
    modifier onlyHook() {
        require(msg.sender == hookContract, "Only hook can call");
        _;
    }

    constructor(
        string memory name,
        string memory symbol
    ) ERC20(name, symbol) Ownable(msg.sender) {}
    
    /**
     * @notice Set the hook contract address
     * @param _hookContract Address of the hook contract
     */
    function setHookContract(address _hookContract) external onlyOwner {
        hookContract = _hookContract;
    }
    
    /**
     * @notice Process liquidity addition
     * @param user Address of the user adding liquidity
     * @param delta BalanceDelta from the liquidity operation
     * @param token0 Address of token0
     * @param token1 Address of token1
     */
    function processLiquidityAdded(
        address user,
        BalanceDelta delta,
        address token0,
        address token1
    ) external onlyHook {
        // Award points for adding liquidity
        uint256 points = 10; // Simple example: 10 points per liquidity add
        userPoints[user] += points;
        totalPoints += points;
        
        // Mint tokens to reward the user
        _mint(user, points);
        
        // Extract token amounts from the BalanceDelta
        (int256 amount0, int256 amount1) = _extractAmounts(delta);
        
        // Emit the event with relevant information
        emit LiquidityAdded(
            user,
            points,
            userPoints[user],
            amount0,
            amount1,
            token0,
            token1,
            block.timestamp
        );
    }
    
    /**
     * @notice Process liquidity removal
     * @param user Address of the user removing liquidity
     * @param delta BalanceDelta from the liquidity operation
     * @param token0 Address of token0
     * @param token1 Address of token1
     */
    function processLiquidityRemoved(
        address user,
        BalanceDelta delta,
        address token0,
        address token1
    ) external onlyHook {
        // Award points for removing liquidity
        uint256 points = 5; // Simple example: 5 points per liquidity removal
        userPoints[user] += points;
        totalPoints += points;
        
        // Mint tokens to reward the user
        _mint(user, points);
        
        // Extract token amounts from the BalanceDelta
        (int256 amount0, int256 amount1) = _extractAmounts(delta);
        
        // Emit the event with relevant information
        emit LiquidityRemoved(
            user,
            points,
            userPoints[user],
            amount0,
            amount1,
            token0,
            token1,
            block.timestamp
        );
    }
    
    /**
     * @notice Process swap execution
     * @param user Address of the user performing the swap
     * @param delta BalanceDelta from the swap operation
     * @param token0 Address of token0
     * @param token1 Address of token1
     */
    function processSwap(
        address user,
        BalanceDelta delta,
        address token0,
        address token1
    ) external onlyHook {
        // Award points for trading
        uint256 points = 5; // Simple example: 5 points per swap
        userPoints[user] += points;
        totalPoints += points;
        
        // Mint tokens to reward the user
        _mint(user, points);
        
        // Extract token amounts from the BalanceDelta
        (int256 amount0, int256 amount1) = _extractAmounts(delta);
        
        // Emit the event with relevant information
        emit SwapExecuted(
            user,
            points,
            userPoints[user],
            amount0,
            amount1,
            token0,
            token1,
            block.timestamp
        );
    }
    
    /**
     * @notice Extract token amounts from BalanceDelta
     * @param delta The BalanceDelta to extract from
     * @return amount0 Amount of token0
     * @return amount1 Amount of token1
     */
    function _extractAmounts(BalanceDelta delta) internal pure returns (int256 amount0, int256 amount1) {
        int256 packedValue = BalanceDelta.unwrap(delta);
        amount0 = int128(int256(packedValue));
        amount1 = int128(int256(packedValue >> 128));
        return (amount0, amount1);
    }
} 