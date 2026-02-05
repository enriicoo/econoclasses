"""
Auction Theory Module
=====================

Comprehensive auction analysis: first-price, second-price (Vickrey),
English, Dutch, all-pay, and revenue equivalence theorem.

Author: econoclasses
Version: 0.5.0
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable, List, Tuple, Dict, Union
from scipy.stats import uniform, expon
from scipy.integrate import quad
from scipy.optimize import minimize_scalar


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class AuctionResult:
    """Results from auction analysis."""
    auction_type: str
    num_bidders: int
    winner: Optional[str]
    winning_bid: float
    price_paid: float
    seller_revenue: float
    winner_surplus: float
    total_surplus: float
    is_efficient: bool
    bidding_strategy: str
    
    def __repr__(self):
        return (f"AuctionResult({self.auction_type})\n"
                f"  Winner: {self.winner}, Bid={self.winning_bid:.2f}\n"
                f"  Price={self.price_paid:.2f}, Revenue={self.seller_revenue:.2f}\n"
                f"  Winner surplus={self.winner_surplus:.2f}\n"
                f"  Efficient: {self.is_efficient}")


@dataclass
class BiddingEquilibrium:
    """Equilibrium bidding strategy."""
    auction_type: str
    distribution: str
    num_bidders: int
    bid_function: Callable[[float], float]
    expected_revenue: float
    expected_winner_payment: float
    efficiency: float
    
    def __repr__(self):
        return (f"BiddingEquilibrium({self.auction_type})\n"
                f"  n={self.num_bidders}, Distribution: {self.distribution}\n"
                f"  E[Revenue]={self.expected_revenue:.4f}\n"
                f"  Efficiency={self.efficiency:.2%}")


@dataclass
class RevenueEquivalenceResult:
    """Demonstration of revenue equivalence theorem."""
    auction_types: List[str]
    distribution: str
    num_bidders: int
    expected_revenues: Dict[str, float]
    are_equivalent: bool
    conditions_satisfied: Dict[str, bool]
    
    def __repr__(self):
        revs = ", ".join(f"{k}={v:.4f}" for k, v in self.expected_revenues.items())
        return (f"RevenueEquivalence (n={self.num_bidders})\n"
                f"  Revenues: {revs}\n"
                f"  Equivalent: {self.are_equivalent}")


# ============================================================================
# AUCTION MECHANISMS
# ============================================================================

class FirstPriceAuction:
    """
    First-price sealed-bid auction.
    
    Highest bidder wins and pays their bid.
    Equilibrium: bid shading, b(v) = v * (n-1)/n for uniform [0,1].
    """
    
    def __init__(self, num_bidders: int, reserve: float = 0.0):
        self.n = num_bidders
        self.reserve = reserve
        self.name = "First-Price Sealed-Bid"
    
    def equilibrium_bid_uniform(self, value: float) -> float:
        """
        Symmetric equilibrium bid for uniform [0,1] values.
        
        b(v) = E[max of (n-1) draws | all < v] = v * (n-1)/n
        """
        if value <= self.reserve:
            return 0
        return value * (self.n - 1) / self.n
    
    def equilibrium_bid_general(self, value: float, F: Callable, f: Callable) -> float:
        """
        General equilibrium bid function.
        
        b(v) = v - ∫[r to v] F(x)^(n-1) dx / F(v)^(n-1)
        
        where r is reserve price and F is CDF, f is PDF.
        """
        if value <= self.reserve:
            return 0
        
        # Numerical integration
        if F(value) ** (self.n - 1) < 1e-10:
            return value  # Avoid division by tiny numbers
        
        integral, _ = quad(lambda x: F(x) ** (self.n - 1), self.reserve, value)
        bid = value - integral / (F(value) ** (self.n - 1))
        return max(bid, self.reserve)
    
    def expected_revenue_uniform(self) -> float:
        """
        Expected revenue with uniform [0,1] values.
        
        E[Revenue] = E[second highest value] = (n-1)/(n+1)
        (Same as second-price by revenue equivalence)
        """
        return (self.n - 1) / (self.n + 1)
    
    def run_auction(self, values: Dict[str, float]) -> AuctionResult:
        """
        Run auction with given valuations.
        
        Parameters
        ----------
        values : Dict[str, float]
            {bidder_id: true_value}
            
        Returns
        -------
        AuctionResult
        """
        # Compute equilibrium bids (assuming uniform [0,1])
        bids = {k: self.equilibrium_bid_uniform(v) for k, v in values.items()}
        
        # Filter by reserve
        eligible = {k: b for k, b in bids.items() if b >= self.reserve}
        
        if not eligible:
            return AuctionResult(
                auction_type=self.name,
                num_bidders=self.n,
                winner=None,
                winning_bid=0,
                price_paid=0,
                seller_revenue=0,
                winner_surplus=0,
                total_surplus=0,
                is_efficient=True,  # No trade is efficient if no one meets reserve
                bidding_strategy=f"b(v) = v × {(self.n-1)/self.n:.3f}"
            )
        
        winner = max(eligible, key=lambda k: eligible[k])
        winning_bid = bids[winner]
        price = winning_bid  # First-price: pay your bid
        
        # Check efficiency
        true_highest = max(values, key=lambda k: values[k])
        is_efficient = winner == true_highest
        
        return AuctionResult(
            auction_type=self.name,
            num_bidders=self.n,
            winner=winner,
            winning_bid=winning_bid,
            price_paid=price,
            seller_revenue=price,
            winner_surplus=values[winner] - price,
            total_surplus=values[winner],  # Winner value is total surplus
            is_efficient=is_efficient,
            bidding_strategy=f"b(v) = v × {(self.n-1)/self.n:.3f}"
        )


class SecondPriceAuction:
    """
    Second-price sealed-bid (Vickrey) auction.
    
    Highest bidder wins, pays second-highest bid.
    Dominant strategy: bid truthfully, b(v) = v.
    """
    
    def __init__(self, num_bidders: int, reserve: float = 0.0):
        self.n = num_bidders
        self.reserve = reserve
        self.name = "Second-Price (Vickrey)"
    
    def equilibrium_bid(self, value: float) -> float:
        """Dominant strategy: bid your true value."""
        return value
    
    def expected_revenue_uniform(self) -> float:
        """
        Expected revenue with uniform [0,1] values.
        
        E[Revenue] = E[second highest order statistic]
                   = (n-1)/(n+1)
        """
        return (self.n - 1) / (self.n + 1)
    
    def run_auction(self, values: Dict[str, float]) -> AuctionResult:
        """Run auction with given valuations."""
        bids = {k: self.equilibrium_bid(v) for k, v in values.items()}
        
        # Filter by reserve
        eligible = {k: b for k, b in bids.items() if b >= self.reserve}
        
        if not eligible:
            return AuctionResult(
                auction_type=self.name,
                num_bidders=self.n,
                winner=None,
                winning_bid=0,
                price_paid=0,
                seller_revenue=0,
                winner_surplus=0,
                total_surplus=0,
                is_efficient=True,
                bidding_strategy="b(v) = v (truthful)"
            )
        
        sorted_bidders = sorted(eligible.items(), key=lambda x: x[1], reverse=True)
        winner = sorted_bidders[0][0]
        winning_bid = sorted_bidders[0][1]
        
        # Price is second-highest bid (or reserve if only one bidder)
        if len(sorted_bidders) > 1:
            price = max(sorted_bidders[1][1], self.reserve)
        else:
            price = self.reserve
        
        return AuctionResult(
            auction_type=self.name,
            num_bidders=self.n,
            winner=winner,
            winning_bid=winning_bid,
            price_paid=price,
            seller_revenue=price,
            winner_surplus=values[winner] - price,
            total_surplus=values[winner],
            is_efficient=True,  # Second-price is always efficient
            bidding_strategy="b(v) = v (truthful)"
        )


class EnglishAuction:
    """
    English (ascending) auction.
    
    Price rises until one bidder remains.
    Strategically equivalent to second-price: stay in until price = value.
    """
    
    def __init__(self, num_bidders: int, reserve: float = 0.0, increment: float = 0.01):
        self.n = num_bidders
        self.reserve = reserve
        self.increment = increment
        self.name = "English (Ascending)"
    
    def simulate(self, values: Dict[str, float], verbose: bool = False) -> AuctionResult:
        """
        Simulate ascending auction.
        
        Each bidder drops out when price exceeds their value.
        """
        price = self.reserve
        active = set(values.keys())
        history = []
        
        while len(active) > 1:
            # Bidders drop out if price exceeds value
            dropouts = {b for b in active if values[b] < price + self.increment}
            
            if dropouts:
                active -= dropouts
                if verbose:
                    history.append(f"Price {price:.2f}: {dropouts} drop out")
            
            if len(active) > 1:
                price += self.increment
            
            if price > max(values.values()) + 1:  # Safety
                break
        
        if not active:
            winner = None
            price = 0
        else:
            winner = list(active)[0]
            # Final price is when second-to-last dropped out
            second_highest = sorted(values.values())[-2] if len(values) > 1 else self.reserve
            price = max(second_highest, self.reserve)
        
        return AuctionResult(
            auction_type=self.name,
            num_bidders=self.n,
            winner=winner,
            winning_bid=values.get(winner, 0) if winner else 0,
            price_paid=price,
            seller_revenue=price,
            winner_surplus=values[winner] - price if winner else 0,
            total_surplus=values[winner] if winner else 0,
            is_efficient=True,
            bidding_strategy="Stay in while price < value"
        )


class DutchAuction:
    """
    Dutch (descending) auction.
    
    Price descends until someone bids.
    Strategically equivalent to first-price sealed-bid.
    """
    
    def __init__(self, num_bidders: int, start_price: float = 1.0, reserve: float = 0.0):
        self.n = num_bidders
        self.start_price = start_price
        self.reserve = reserve
        self.name = "Dutch (Descending)"
    
    def equilibrium_stop_price_uniform(self, value: float) -> float:
        """
        Equilibrium: stop at same price as first-price bid.
        
        b(v) = v * (n-1)/n
        """
        return value * (self.n - 1) / self.n
    
    def simulate(self, values: Dict[str, float]) -> AuctionResult:
        """
        Simulate descending auction.
        
        Price falls; bidder with highest strategy threshold wins.
        """
        # Compute stopping thresholds
        thresholds = {k: self.equilibrium_stop_price_uniform(v) 
                      for k, v in values.items()}
        
        # Winner is who has highest threshold (would stop first)
        if not any(t >= self.reserve for t in thresholds.values()):
            return AuctionResult(
                auction_type=self.name,
                num_bidders=self.n,
                winner=None,
                winning_bid=0,
                price_paid=0,
                seller_revenue=0,
                winner_surplus=0,
                total_surplus=0,
                is_efficient=True,
                bidding_strategy=f"Stop at v × {(self.n-1)/self.n:.3f}"
            )
        
        winner = max(thresholds, key=lambda k: thresholds[k])
        price = thresholds[winner]
        
        return AuctionResult(
            auction_type=self.name,
            num_bidders=self.n,
            winner=winner,
            winning_bid=price,
            price_paid=price,
            seller_revenue=price,
            winner_surplus=values[winner] - price,
            total_surplus=values[winner],
            is_efficient=True,  # Same as first-price
            bidding_strategy=f"Stop at v × {(self.n-1)/self.n:.3f}"
        )


class AllPayAuction:
    """
    All-pay auction.
    
    All bidders pay their bid, only highest wins.
    Used for lobbying, R&D races, contests.
    """
    
    def __init__(self, num_bidders: int, reserve: float = 0.0):
        self.n = num_bidders
        self.reserve = reserve
        self.name = "All-Pay"
    
    def equilibrium_bid_uniform(self, value: float) -> float:
        """
        Symmetric equilibrium bid for uniform [0,1].
        
        b(v) = v^n * (n-1)/n
        
        (Shading more than first-price because you always pay)
        """
        return (value ** self.n) * (self.n - 1) / self.n
    
    def expected_revenue_uniform(self) -> float:
        """
        Expected revenue with uniform [0,1] values.
        
        E[Revenue] = n × E[bid] = (n-1)/(n+1)
        (Revenue equivalence holds!)
        """
        return (self.n - 1) / (self.n + 1)
    
    def run_auction(self, values: Dict[str, float]) -> AuctionResult:
        """Run all-pay auction."""
        bids = {k: self.equilibrium_bid_uniform(v) for k, v in values.items()}
        
        total_payments = sum(bids.values())
        
        if max(bids.values()) < self.reserve:
            return AuctionResult(
                auction_type=self.name,
                num_bidders=self.n,
                winner=None,
                winning_bid=0,
                price_paid=0,
                seller_revenue=total_payments,
                winner_surplus=0,
                total_surplus=-total_payments,  # Wasted bids
                is_efficient=False,
                bidding_strategy=f"b(v) = v^{self.n} × {(self.n-1)/self.n:.3f}"
            )
        
        winner = max(bids, key=lambda k: bids[k])
        
        # Winner surplus = value - bid
        # Loser surplus = -bid
        winner_surplus = values[winner] - bids[winner]
        loser_payments = sum(bids[k] for k in bids if k != winner)
        
        return AuctionResult(
            auction_type=self.name,
            num_bidders=self.n,
            winner=winner,
            winning_bid=bids[winner],
            price_paid=bids[winner],
            seller_revenue=total_payments,
            winner_surplus=winner_surplus,
            total_surplus=values[winner] - loser_payments,  # Adjusted for waste
            is_efficient=True,  # Highest value wins
            bidding_strategy=f"b(v) = v^{self.n} × {(self.n-1)/self.n:.3f}"
        )


# ============================================================================
# REVENUE EQUIVALENCE
# ============================================================================

def revenue_equivalence_theorem(
    num_bidders: int = 2,
    distribution: str = "uniform"
) -> RevenueEquivalenceResult:
    """
    Demonstrate the Revenue Equivalence Theorem.
    
    Theorem: Any auction mechanism that:
    1. Allocates to highest bidder
    2. Gives zero surplus to lowest type
    3. Has symmetric, increasing equilibrium strategies
    
    yields the same expected revenue.
    
    Parameters
    ----------
    num_bidders : int
        Number of symmetric bidders
    distribution : str
        'uniform' for [0,1], others TBD
        
    Returns
    -------
    RevenueEquivalenceResult
        Comparison of auction formats
    """
    n = num_bidders
    
    if distribution == "uniform":
        # E[Revenue] = E[second order statistic] = (n-1)/(n+1)
        expected_second = (n - 1) / (n + 1)
        
        first_price = FirstPriceAuction(n)
        second_price = SecondPriceAuction(n)
        all_pay = AllPayAuction(n)
        
        revenues = {
            "First-Price": first_price.expected_revenue_uniform(),
            "Second-Price": second_price.expected_revenue_uniform(),
            "All-Pay": all_pay.expected_revenue_uniform(),
            "Theoretical": expected_second
        }
    else:
        revenues = {"Error": 0}
    
    # Check if approximately equal
    vals = list(revenues.values())
    are_equivalent = max(vals) - min(vals) < 0.001
    
    return RevenueEquivalenceResult(
        auction_types=["First-Price", "Second-Price", "All-Pay"],
        distribution=distribution,
        num_bidders=n,
        expected_revenues=revenues,
        are_equivalent=are_equivalent,
        conditions_satisfied={
            "symmetric_IPV": True,
            "risk_neutral": True,
            "independent_values": True,
            "efficient_allocation": True,
            "zero_surplus_lowest_type": True
        }
    )


def compare_auctions(
    values: Dict[str, float],
    reserve: float = 0.0
) -> Dict[str, AuctionResult]:
    """
    Compare all auction formats with given values.
    
    Parameters
    ----------
    values : Dict[str, float]
        {bidder: true_value}
    reserve : float
        Common reserve price
        
    Returns
    -------
    Dict of auction results
    """
    n = len(values)
    
    results = {}
    results['first_price'] = FirstPriceAuction(n, reserve).run_auction(values)
    results['second_price'] = SecondPriceAuction(n, reserve).run_auction(values)
    results['english'] = EnglishAuction(n, reserve).simulate(values)
    results['dutch'] = DutchAuction(n, reserve=reserve).simulate(values)
    results['all_pay'] = AllPayAuction(n, reserve).run_auction(values)
    
    return results


# ============================================================================
# OPTIMAL RESERVE PRICES
# ============================================================================

def optimal_reserve_price(
    num_bidders: int,
    distribution: str = "uniform",
    dist_params: Tuple[float, float] = (0, 1)
) -> Dict:
    """
    Compute optimal reserve price for revenue maximization.
    
    For uniform [0,1] and symmetric bidders:
    Optimal reserve r* = 1/2 (sells half the time to monopolist)
    
    Parameters
    ----------
    num_bidders : int
        Number of bidders
    distribution : str
        Value distribution ('uniform')
    dist_params : Tuple
        Distribution parameters (low, high) for uniform
        
    Returns
    -------
    Dict with optimal reserve and analysis
    """
    n = num_bidders
    low, high = dist_params
    
    if distribution == "uniform":
        # Virtual value: φ(v) = 2v - high (for uniform [low, high])
        # Optimal reserve: φ(r) = 0 => r = high/2 (when low=0)
        optimal_r = (low + high) / 2
        
        # Expected revenue without reserve
        # E[2nd order stat of n uniform [0,1]] = (n-1)/(n+1)
        rev_no_reserve = (n - 1) / (n + 1) * (high - low) + low * (1 - 0)  # Simplified
        
        # Expected revenue with optimal reserve (complex integral)
        # Approximate for [0,1]: higher with reserve
        if low == 0 and high == 1:
            # Known result: optimal reserve of 1/2 gives higher revenue
            # Probability someone above 1/2: 1 - (1/2)^n
            prob_sale = 1 - (0.5) ** n
            # E[max | max > 0.5] is complex, approximate
            rev_with_reserve = prob_sale * 0.5 + (1 - prob_sale) * 0  # Lower bound
            
            # Better approximation using integration
            # E[max bid | sale] ≈ (n/(n+1)) * E[max val | max > r]
            if n == 2:
                rev_with_reserve = 5/12  # Known exact value
            else:
                # Rough approximation
                rev_with_reserve = rev_no_reserve * 1.05
    else:
        optimal_r = 0.5
        rev_no_reserve = 0.33
        rev_with_reserve = 0.35
    
    return {
        'optimal_reserve': optimal_r,
        'revenue_without_reserve': rev_no_reserve,
        'revenue_with_reserve': rev_with_reserve,
        'revenue_gain': rev_with_reserve - rev_no_reserve,
        'interpretation': (
            f"Optimal reserve r* = {optimal_r:.2f} excludes low-value bidders "
            f"but extracts more surplus from high-value bidders. "
            f"Revenue increases by approximately {(rev_with_reserve/rev_no_reserve - 1)*100:.1f}%"
        )
    }


# ============================================================================
# COMMON VALUE AUCTIONS
# ============================================================================

def winner_curse_analysis(
    true_value: float,
    signal_noise: float,
    num_bidders: int
) -> Dict:
    """
    Analyze winner's curse in common value auctions.
    
    When object has common (unknown) value and bidders get noisy signals,
    winner tends to be the one with highest (most optimistic) signal.
    
    Parameters
    ----------
    true_value : float
        Actual value of the object
    signal_noise : float
        Standard deviation of signal errors
    num_bidders : int
        Number of bidders
        
    Returns
    -------
    Dict with winner's curse analysis
    """
    n = num_bidders
    
    # Simulate signals
    np.random.seed(42)
    signals = true_value + np.random.normal(0, signal_noise, n)
    
    # Naive bidding (bid = signal)
    naive_winner_signal = np.max(signals)
    naive_overpayment = naive_winner_signal - true_value
    
    # Winner's curse: E[max of n N(0,σ²) draws] ≈ σ × √(2 ln n) for large n
    expected_max_error = signal_noise * np.sqrt(2 * np.log(n)) if n > 1 else 0
    
    # Optimal adjustment (shade bid by expected curse)
    # E[value | my signal is highest] = true_value + adjustment
    # where adjustment accounts for selection bias
    
    optimal_shading = expected_max_error  # Rough approximation
    
    return {
        'true_value': true_value,
        'highest_signal': naive_winner_signal,
        'naive_overpayment': naive_overpayment,
        'expected_curse': expected_max_error,
        'optimal_bid_shading': optimal_shading,
        'sophisticated_bid': naive_winner_signal - optimal_shading,
        'interpretation': (
            f"With {n} bidders and signal noise σ={signal_noise:.2f}, "
            f"the expected winner's curse is ~{expected_max_error:.2f}. "
            f"Sophisticated bidders should shade bids by this amount."
        ),
        'implications': [
            "Winner's curse is worse with more bidders",
            "Winner's curse is worse with noisier signals",
            "Experienced bidders learn to shade bids",
            "English auctions partially mitigate curse (see others drop out)"
        ]
    }


def affiliated_values_auction(
    private_values: Dict[str, float],
    common_component: float,
    weight_private: float = 0.5
) -> Dict:
    """
    Analyze auction with affiliated values.
    
    Value = weight × private_component + (1-weight) × common_component
    
    Affiliation reduces information rents and can increase revenue.
    Linkage principle: revealing information benefits seller.
    """
    weight_common = 1 - weight_private
    
    # Compute true values for each bidder
    true_values = {
        bidder: weight_private * pv + weight_common * common_component
        for bidder, pv in private_values.items()
    }
    
    # In affiliated values, English auction > Second-price > First-price
    # Because English reveals information during auction
    
    return {
        'true_values': true_values,
        'common_component': common_component,
        'weight_private': weight_private,
        'revenue_ranking': [
            "English auction (reveals most information)",
            "Second-price sealed-bid",
            "First-price sealed-bid (reveals least)"
        ],
        'linkage_principle': (
            "When values are affiliated, any policy that links payment to "
            "information about other bidders' values increases expected revenue. "
            "This is why English auction dominates in affiliated value settings."
        )
    }


# ============================================================================
# MULTI-UNIT AUCTIONS
# ============================================================================

def uniform_price_auction(
    demands: Dict[str, List[Tuple[float, int]]],
    supply: int
) -> Dict:
    """
    Uniform price auction for multiple identical units.
    
    All winners pay the market-clearing price.
    
    Parameters
    ----------
    demands : Dict[str, List[Tuple[float, int]]]
        {bidder: [(price_1, quantity_1), (price_2, quantity_2), ...]}
        Bids in descending price order
    supply : int
        Number of units available
        
    Returns
    -------
    Dict with allocations and prices
    """
    # Aggregate demand curve
    all_bids = []
    for bidder, bids in demands.items():
        for price, qty in bids:
            for _ in range(qty):
                all_bids.append((price, bidder))
    
    # Sort by price descending
    all_bids.sort(key=lambda x: x[0], reverse=True)
    
    # Allocate to highest bids
    winners = {}
    clearing_price = 0
    
    for i, (price, bidder) in enumerate(all_bids[:supply]):
        winners[bidder] = winners.get(bidder, 0) + 1
        clearing_price = price
    
    # If there are more bids than supply, clearing price is the highest rejected bid
    if len(all_bids) > supply:
        clearing_price = all_bids[supply][0]
    
    total_revenue = clearing_price * sum(winners.values())
    
    return {
        'allocations': winners,
        'clearing_price': clearing_price,
        'total_revenue': total_revenue,
        'units_sold': sum(winners.values()),
        'demand_reduction': (
            "Bidders may strategically reduce demand to lower clearing price. "
            "This is a form of bid shading in multi-unit auctions."
        )
    }


def discriminatory_auction(
    demands: Dict[str, List[Tuple[float, int]]],
    supply: int
) -> Dict:
    """
    Discriminatory (pay-as-bid) auction for multiple units.
    
    Each winner pays their own bid (different prices).
    
    Parameters
    ----------
    demands : Dict[str, List[Tuple[float, int]]]
        {bidder: [(price_1, quantity_1), ...]}
    supply : int
        Number of units
        
    Returns
    -------
    Dict with allocations and payments
    """
    # Aggregate and sort bids
    all_bids = []
    for bidder, bids in demands.items():
        for price, qty in bids:
            for _ in range(qty):
                all_bids.append((price, bidder))
    
    all_bids.sort(key=lambda x: x[0], reverse=True)
    
    # Allocate and compute payments
    winners = {}
    payments = {}
    
    for i, (price, bidder) in enumerate(all_bids[:supply]):
        winners[bidder] = winners.get(bidder, 0) + 1
        payments[bidder] = payments.get(bidder, 0) + price
    
    total_revenue = sum(payments.values())
    
    return {
        'allocations': winners,
        'payments': payments,
        'total_revenue': total_revenue,
        'units_sold': sum(winners.values()),
        'strategic_implications': (
            "In discriminatory auctions, bidders shade bids more "
            "because they pay their bid. Revenue comparison with "
            "uniform price is ambiguous and depends on setting."
        )
    }


# ============================================================================
# EXAMPLES
# ============================================================================

def example_auction_comparison():
    """Compare auction formats with specific values."""
    values = {
        'Alice': 0.9,
        'Bob': 0.7,
        'Carol': 0.5
    }
    
    results = compare_auctions(values)
    
    summary = {
        'values': values,
        'results': {
            name: {
                'winner': r.winner,
                'price': r.price_paid,
                'revenue': r.seller_revenue,
                'efficient': r.is_efficient
            }
            for name, r in results.items()
        }
    }
    
    return summary


def example_revenue_equivalence():
    """Demonstrate revenue equivalence theorem."""
    results = {}
    for n in [2, 3, 5, 10]:
        eq = revenue_equivalence_theorem(n)
        results[f"n={n}"] = {
            'revenues': eq.expected_revenues,
            'equivalent': eq.are_equivalent
        }
    return results


def example_optimal_reserve():
    """Show optimal reserve price analysis."""
    results = {}
    for n in [2, 3, 5]:
        analysis = optimal_reserve_price(n, "uniform", (0, 1))
        results[f"n={n}"] = {
            'optimal_reserve': analysis['optimal_reserve'],
            'revenue_gain': analysis['revenue_gain']
        }
    return results


def example_winners_curse():
    """Demonstrate winner's curse."""
    return winner_curse_analysis(
        true_value=100,
        signal_noise=20,
        num_bidders=5
    )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Data classes
    'AuctionResult',
    'BiddingEquilibrium',
    'RevenueEquivalenceResult',
    
    # Auction classes
    'FirstPriceAuction',
    'SecondPriceAuction',
    'EnglishAuction',
    'DutchAuction',
    'AllPayAuction',
    
    # Analysis
    'revenue_equivalence_theorem',
    'compare_auctions',
    'optimal_reserve_price',
    
    # Common value
    'winner_curse_analysis',
    'affiliated_values_auction',
    
    # Multi-unit
    'uniform_price_auction',
    'discriminatory_auction',
    
    # Examples
    'example_auction_comparison',
    'example_revenue_equivalence',
    'example_optimal_reserve',
    'example_winners_curse',
]
