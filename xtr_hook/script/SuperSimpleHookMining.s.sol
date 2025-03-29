// NOTE: This is based on V4PreDeployed.s.sol
// You can make changes to base on V4Deployer.s.sol to deploy everything fresh as well

// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import {Script} from "forge-std/Script.sol";
import {PoolManager} from "v4-core/PoolManager.sol";
import {PoolKey} from "v4-core/types/PoolKey.sol";
import {Hooks} from "v4-core/libraries/Hooks.sol";
import {HookMiner} from "../test/HookMiner.sol";
import {BasicHook} from "../src/BasicHook.sol";
import "forge-std/console.sol";

contract HookMiningSample is Script {
    PoolManager manager =
        PoolManager(0xf2873F92324E8EC98a82C47AFA0e728Bd8E41665);

    function setUp() public {
        uint160 flags = uint160(Hooks.BEFORE_SWAP_FLAG);
        address CREATE2_DEPLOYER = 0x4e59b44847b379578588920cA78FbF26c0B4956C;
        (address hookAddress, bytes32 salt) = HookMiner.find(
            CREATE2_DEPLOYER,
            flags,
            type(BasicHook).creationCode,
            abi.encode(address(manager))
        );

        vm.startBroadcast();
        BasicHook hook = new BasicHook{salt: salt}(manager);
        require(address(hook) == hookAddress, "hook address mismatch");
        vm.stopBroadcast();
    }

    function run() public pure {
        console.log("Hello");
    }
}
