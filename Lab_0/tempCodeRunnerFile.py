 while la.norm((yn - y), np.inf) >= epsilon:
        x = la.solve(A, y)
        y = yn
        yn = np.dot((1/la.norm(x, np.inf)), x)
