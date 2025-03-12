# Uniswap V4 LP Hedging Strategy Analysis

This repository contains the LaTeX documentation and Python implementation of hedging strategies for Uniswap V4 liquidity positions using perpetual futures.

## Setup and Usage

Requirements:
- Python 3.8+
- LaTeX distribution (e.g., TeX Live, MiKTeX)
- Python packages: numpy, matplotlib

Installation:
1. Clone this repository
2. Set up Python environment:
   cd pythonScripts
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt

Generate Plots:
cd pythonScripts
chmod +x plotAll.sh  # Make script executable
./plotAll.sh        # Generate all plots

Compile LaTeX:
pdflatex main.tex

## Components

Python Classes:

UniswapConcentratedLiquidityPosition
- Implements Uniswap V4 concentrated liquidity mechanics
- Calculates position value across price ranges
- Handles transitions at range boundaries

LongPerpPayoff & ShortPerpPayoff
- Calculate perpetual futures payoffs
- Handle liquidation boundaries
- Account for margin requirements

CombinedPosition Classes
- Merge LP and perpetual positions
- Implement different hedging strategies
- Calculate combined payoffs

LaTeX Document:
- Theoretical framework for Uniswap V4 concentrated liquidity
- Analysis of perpetual futures hedging strategies
- Derivation of combined payoff functions
- Visualization and interpretation of results

## Key Features

1. Piecewise Payoff Calculations
- Accurate modeling of LP position transitions
- Perpetual futures with liquidation boundaries
- Combined strategy analysis

2. Risk Management
- Liquidation price computations
- Capital efficiency considerations
- Hedge ratio optimization

3. Visualization
- Individual component payoffs
- Combined strategy results
- Multiple hedging scenarios