// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import {Script} from "forge-std/Script.sol";
import {PoolManager} from "v4-core/PoolManager.sol";
import {Hooks} from "v4-core/libraries/Hooks.sol";
import {HookMiner} from "../test/HookMiner.sol";
import {xtrHook} from "../src/xtrHook.sol";
import {xtrLogic} from "../src/xtrLogic.sol";
import "forge-std/console.sol";

contract DeployXtrHook is Script {
    // Address of PoolManager deployed on Arbitrum
    PoolManager manager =
        PoolManager(0x360E68faCcca8cA495c1B759Fd9EEe466db9FB32);

    function setUp() public {
        vm.startBroadcast();
        
        // 1. Deploy the xtrLogic contract first
        xtrLogic logic = new xtrLogic("xtrPoints", "XTP");
        console.log("Deployed xtrLogic at", address(logic));
        
        // 2. Extract the flags based on xtrHook's getHookPermissions function
        uint160 flags = uint160(
            Hooks.AFTER_ADD_LIQUIDITY_FLAG | 
            Hooks.AFTER_REMOVE_LIQUIDITY_FLAG |
            Hooks.AFTER_SWAP_FLAG
        );

        // 3. Mine for a hook address
        address CREATE2_DEPLOYER = 0x4e59b44847b379578588920cA78FbF26c0B4956C;
        (address hookAddress, bytes32 salt) = HookMiner.find(
            CREATE2_DEPLOYER,
            flags,
            type(xtrHook).creationCode,
            abi.encode(address(manager), address(logic))
        );

        // 4. Deploy the hook with the found salt
        xtrHook hook = new xtrHook{salt: salt}(manager, address(logic));
        require(address(hook) == hookAddress, "hook address mismatch");
        console.log("Deployed xtrHook at", address(hook));
        
        // 5. Set the hook address in the logic contract
        logic.setHookContract(address(hook));
        console.log("Set hook contract in logic contract");
        
        vm.stopBroadcast();
    }

    function run() public {
        setUp();
    }
} 