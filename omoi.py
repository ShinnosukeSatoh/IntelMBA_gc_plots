import time

k = 0


def ack(m, n):
    for i in range(n):
        for j in range(m):
            k = m + n

    return 0


def main():
    start = time.time()

    ack(20000, 20000)

    end = time.time()

    print('END [sec]: ', end-start)

    duration = end - start

    return duration


if __name__ == '__main__':
    d = 0
    for i in range(30):
        d += main()

    print('AVERAGE [sec]: ', d/30)
