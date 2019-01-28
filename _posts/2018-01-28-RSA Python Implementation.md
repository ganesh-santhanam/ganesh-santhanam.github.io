---
title: "RSA Encryption implementation in Python"
mathjax: "true"
---

# RSA

This is a explanation of RSA encryption, along with a simple Python implementation of it.


RSA (Rivest–Shamir–Adleman) is one of the first public-key cryptosystems. The encryption key is public and it is different from the decryption key which is kept secret (private). In RSA, this asymmetry is based on the practical difficulty of the factorization of the product of two large prime numbers, the "factoring problem". The acronym RSA is made of the initial letters of the surnames of Ron Rivest, Adi Shamir, and Leonard Adleman, who first publicly described the algorithm in 1978

It relies on the following principles:
* It is easy to find two distinct large primes pp and q.q.
* It is easy to multiply two large primes together, calculating n=pq.n=pq.
* However, given n=pq,n=pq, it is difficult to factor out pp and q.q.
Given a, n and e , with 0<a<n and e > 1 calculating $a ^ { e } ( \mathrm { mod } n )$ is easy.
* But inversely, given $a ^ { e } ( \mathrm { mod } n )$ ,e and n, it is difficult to calculate a.
* However, if the multiplicative inverse of e modulo $\phi ( n ) = ( q - 1 ) ( p - 1 )$ labelled d is given, then the previous calculation is easy.
* Calculating d from only n and e is difficult, but easy if the factorization of n=pq is known.

# The steps for the algorithm is as follows:  

## Key Generation:
Generate distinct primes p and q and let n=pq. Also let $m=\phi(n)=(p-1)(q-1)$  and pick any 1<e<m. Calculate d as the multiplicative inverse of e modulo m using the extended Euclidean algorithm. The private key is then (n, d) and the public key is (n, e).  

## Encryption

To encrypt plaintext, first encode the plaintext (by padding it for example) so that it consists of blocks of size less than n. Then for block a, define $E(a)=a^e \pmod{n}$ as the encryption function.  

## Decryption:
To decrypt ciphertext given by c = E(a) define $D(c) = c^d \pmod{n}$. We then have $D(c)=D(E(a))=a^{ed} \pmod{n}$. For this to work, we need $a^{ed}=a \pmod{n}$  


# Key Generation and Primality Tests

To start, our key generation requires us to generate two large primes p and q.
To generate a large prime p,p, we start by deciding the number of bits we require in the prime. Let's call this number b and let's assume for simplicity that b>8 for all our intended purposes. In practice, b is usually between 512 and 2048.
Then we proceed to check t+it+i for primality starting from i=0i=0 and onwards.

We use the Rabin-Miller algorithms to check if the number is prime or not. Then implement the RSA algorithm using the steps mentioned above

<p align="center">
<img src="https://imgur.com/FKRE08g.jpg">

</p>

<center>
RSA Algorithm
</center>


```python
def generate_random_prime(bits, primality_test):
    """Generate random prime number with n bits."""
    get_random_t = lambda: random.getrandbits(bits) | 1 << bits | 1
    p = get_random_t()
    for i in itertools.count(1):
        if primality_test(p):
            return p
        else:
            if i % (bits * 2) == 0:
                p = get_random_t()
            else:
                p += 2  # Add 2 since we are only interested in odd numbers

                @logged("%b %d %Y - %H:%M:%S")
                def simple_is_prime(n):
                    """Returns True if n is a prime. False otherwise."""
                    if n % 2 == 0:
                        return n == 2
                    if n % 3 == 0:
                        return n == 3
                    k = 6L
                    while (k - 1) ** 2 <= n:
                        if n % (k - 1) == 0 or n % (k + 1) == 0:
                            return False
                        k += 6
                    return True


                    def rabin_miller_is_prime(n, k=20):
                        """
                        Test n for primality using Rabin-Miller algorithm, with k
                        random witness attempts. False return means n is certainly a composite.
                        True return value indicates n is *probably* a prime. False positive
                        probability is reduced exponentially the larger k gets.
                        """
                        b = basic_is_prime(n, K=100)
                        if b is not None:
                            return b

                        m = n - 1
                        s = 0
                        while m % 2 == 0:
                            s += 1
                            m //= 2
                        liars = set()
                        get_new_x = lambda: random.randint(2, n - 1)
                        while len(liars) < k:
                            x = get_new_x()
                            while x in liars:
                                x = get_new_x()
                            xi = pow(x, m, n)
                            witness = True
                            if xi == 1 or xi == n - 1:
                                witness = False
                            else:
                                for __ in xrange(s - 1):
                                    xi = (xi ** 2) % n
                                    if xi == 1:
                                        return False
                                    elif xi == n - 1:
                                        witness = False
                                        break
                                xi = (xi ** 2) % n
                                if xi != 1:
                                    return False
                            if witness:
                                return False
                            else:
                                liars.add(x)
                        return True                  

 def basic_is_prime(n, K=-1):
    """Returns True if n is a prime, and False it is a composite
    by trying to divide it by two and first K odd primes. Returns
    None if test is inconclusive."""
    if n % 2 == 0:
        return n == 2
    for p in primes_list.less_than_hundred_thousand[:K]:
        if n % p == 0:
            return n == p
 return None                


 def power(x, m, n):
    """Calculate x^m modulo n using O(log(m)) operations."""
    a = 1
    while m > 0:
        if m % 2 == 1:
            a = (a * x) % n
        x = (x * x) % n
        m //= 2
    return a

    def extended_gcd(a, b):
        """Returns pair (x, y) such that xa + yb = gcd(a, b)"""
        x, lastx, y, lasty = 0, 1, 1, 0
        while b != 0:
            q, r = divmod(a, b)
            a, b = b, r
            x, lastx = lastx - q * x, x
            y, lasty = lasty - q * y, y
        return lastx, lasty


    def multiplicative_inverse(e, n):
        """Find the multiplicative inverse of e mod n."""
        x, y = extended_gcd(e, n)
        if x < 0:
            return n + x
        return x


    def rsa_generate_key(bits):
        p = generate_random_prime(bits / 2)
        q = generate_random_prime(bits / 2)
        # Ensure q != p, though for large values of bits this is
        # statistically very unlikely
        while q == p:
            q = generate_random_prime(bits / 2)
        n = p * q
        phi = (p - 1) * (q - 1)
        # Here we pick a random e, but a fixed value for e can also be used.
        while True:
            e = random.randint(3, phi - 1)
            if fractions.gcd(e, phi) == 1:
                break
        d = multiplicative_inverse(e, phi)
        return (n, e, d)


    def rsa_encrypt(message, n, e):
        return modular.power(message, e, n)


    def rsa_decrypt(cipher, n, d):
        return modular.power(cipher, d, n)




```
