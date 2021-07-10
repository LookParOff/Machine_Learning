import random


class Hand:
    def __init__(self, maxReward):
        self.maxReward = maxReward

    def pull(self):
        return random.random() * self.maxReward


class Bandit:
    def __init__(self, rewardsForHands):
        # rewardsForHands[i] maximum reward for i hand of bandit
        self.hands = []
        for r in rewardsForHands:
            self.hands.append(Hand(r))


def chooseRandomHand(pi):
    # epsilon- prob of choose do not optimal hand. k- count of hand
    temp = random.random()
    indexOfMax = pi.index(max(pi))
    if max(pi) < temp:
        # choose randomly not optimal hand. Exploration
        index = random.randint(0, len(pi) - 1)
        if index == indexOfMax:
            index -= 1
        return index
    else:
        # choose optimal hand. Exploitation
        return indexOfMax


def findOptimHand(bandit, eps):
    k = len(bandit.hands)
    estimations = [100 for _ in range(k)]
    n = [1 for _ in range(k)]  # n[i]- how much pull the i hand
    pi = [1 / k for _ in range(k)]  # pi[i] is probability that i hand is optimal
    j = 0
    while j < 1000:
        indexOfHand = chooseRandomHand(pi)
        hand = bandit.hands[indexOfHand]
        r = hand.pull()
        estimations[indexOfHand] = estimations[indexOfHand] + 1 / n[indexOfHand] * (r - estimations[indexOfHand])
        n[indexOfHand] += 1
        if pi.index(max(pi)) != estimations.index(max(estimations)):
            # estimation had changed now fix probabilities
            pi = [eps / k] * k
            pi[estimations.index(max(estimations))] = (1 - eps) + eps / k
        j += 1
    print(estimations)
    return estimations.index(max(estimations))


def main():
    rewardsForHands = [37, 15.5, 30, 21, 10, 22, 1, 36.9]
    bandit = Bandit(rewardsForHands)
    optimHand = findOptimHand(bandit, 0.1)
    print(optimHand)


if __name__ == "__main__":
    main()

