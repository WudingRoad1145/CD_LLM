# Multi-sided incomplete information

### Simple version - 2-sided incomplete information
A seller (player S) owns an item that a buyer (player B) would like to purchase. The seller’s reservation price is s (that is, she is willing to sell if and only if the price paid by the buyer is at least s) and the buyer’s reservation price is b (that is, he is willing to buy if and only if the price is less than or equal to b). 
It is common knowledge between the two that
    - both b and s belong to the set {1,2,...,n},
    - the buyer knows the value of b and the seller knows the value of s,
    - both the buyer and the seller attach equal probability to all the possibilities among which they are uncertain.

Buyer and Seller play the following game. First the buyer makes an offer of a price p ∈ {1,...,n} to the seller. If p = n the game ends and the object is exchanged for $p. If p < n then the seller either accepts (in which case the game ends and the object is exchanged for p) or makes a counter-offer of p′ > p, in which case either the buyer accepts (and the game ends and the object is exchanged for p′) or the buyer rejects, in which case the game ends without an exchange.

Payoffs are as follows:
Seller: 
    - 0 if there is no exchange
    - x-s if exchange takes place at price $x
Buyer: 
    - 0 if there is no exchange
    - b-p if exchange takes place at price $p (the initial offer)
    - b-p'-theta if exchange takes place at price $p' (the counter-offer)
            where theta > 0 is a measure of wasted time and potential cost from negotiation

These are von Neumann-Morgenstern payoffs.