class ApproximatingPair:
    def __init__(self, *args):
        if isinstance(args[0], int):
            if len(args) == 3:
                self.initialize_from_epsilon(args[0], args[1], args[2])
            else:
                self.initialize_from_discretization(args[0], args[1])
        else:
            self.initialize_from_data(args[0], args[1], args[2])
        self.a = self.range_array[0]
        self.b = self.range_array[len(self.range_array)-1]
        self.n = len(self.range_array)

    def initialize_from_epsilon(self, n, epsilon, base_distribution):
        x0 = base_distribution.range_()[0]
        xn = base_distribution.range_()[1]
        r = xn - x0
        ra = []
        ua = []
        la = []
        ra.append(x0)
        ua.append(0.0)
        la.append(0.0)
        for i in range(1, n + 1):
            ra.append(x0 + i * r / n)
            exact_value = base_distribution.cdf(x0 + i * r / n)
            # Add or remove small amount
            if exact_value + epsilon < 1:
                ua.append(exact_value + epsilon)
            else:
                ua.append(1)
            if 0 < exact_value - epsilon:
                la.append(exact_value - epsilon)
            else:
                la.append(0)
        ra.append(xn)
        ua.append(1.0)
        la.append(1.0)
        self.range_array = ra
        self.upper_array = ua
        self.lower_array = la

    def initialize_from_discretization(self, n, base_distribution):
        x0 = base_distribution.range_()[0]
        xn = base_distribution.range_()[1]
        r = (xn - x0) / n
        ra = []
        ua = []
        la = []
        for i in range(0, n+1):
            ra.append(x0 + i * r)
            la.append(base_distribution.cdf(ra[i]))
        self.range_array = ra
        self.upper_array = la
        self.lower_array = la

    def initialize_from_data(self, range_array, upper_array, lower_array):
        if len(range_array) != len(upper_array) or len(range_array) != len(lower_array):
            raise ValueError("range_array and value_array must have the same length")
        self.range_array = range_array
        self.upper_array = upper_array
        self.lower_array = lower_array


class IndependentOperation:
    def __init__(self, operation, left_operand, right_operand):
        self.operation = operation
        self.left_operand = left_operand
        self.right_operand = right_operand
        self.output = None
        if self.left_operand.n != self.right_operand.n:
            raise ValueError("Left and right operand must be approximating pairs of the same length.")
        self.n = self.left_operand.n


    def perform_operation(self):
        if self.operation == "+":
            if isinstance(self.left_operand, ApproximatingPair) and isinstance(self.right_operand, ApproximatingPair):
                self._perform_AA_Addition()
        if self.operation == "-":
            if isinstance(self.left_operand, ApproximatingPair) and isinstance(self.right_operand, ApproximatingPair):
                self._perform_AA_Subtraction()

    def _perform_AA_Addition(self):
        ax = self.left_operand.a
        bx = self.left_operand.b
        ay = self.right_operand.a
        by = self.right_operand.b
        # Compute range using interval arithmetic
        a = ax + ay
        b = bx + by
        r = (b - a) / (self.n - 1)
        zk = []
        uzk = []
        lzk = []
        zk.append(a)
        uzk.append(0.0)
        lzk.append(0.0)
        for k in range(1, self.n):
            z = a + (k * r)
            zk.append(z)
            l = 0
            u = 0
            for i in range(1, self.n):
                j = self._l_addition(self.left_operand.range_array[i], z, self.right_operand.range_array)
                l += (self.left_operand.lower_array[i]-self.left_operand.upper_array[i-1]) * self.right_operand.lower_array[j]
                j = self._u_addition(self.left_operand.range_array[i-1], z, self.right_operand.range_array)
                u += (self.left_operand.upper_array[i] - self.left_operand.lower_array[i-1]) * self.right_operand.upper_array[j]
            uzk.append(min(u, 1))
            lzk.append(max(l, 0))
        self.output = ApproximatingPair(zk, uzk, lzk)

    def _u_addition(self, x, z, y_array):
        i = 0
        while i < self.n and x + y_array[i] < z:
            i = i + 1
        if i < self.n:
            return i
        else:
            return self.n - 1

    def _l_addition(self, x, z, y_array):
        i = self.n - 1
        while i >= 0 and z < x + y_array[i]:
            i = i - 1
        if i >= 0:
            return i
        else:
            return 0

    def _perform_AA_Subtraction(self):
        ax = self.left_operand.a
        bx = self.left_operand.b
        ay = self.right_operand.a
        by = self.right_operand.b
        # Compute range using interval arithmetic
        a = ax - by
        b = bx - ay
        r = (b - a) / (self.n - 1)
        zk = []
        uzk = []
        lzk = []
        zk.append(a)
        uzk.append(0.0)
        lzk.append(0.0)
        for k in range(1, self.n):
            z = a + (k * r)
            zk.append(z)
            l = 0
            u = 0
            for i in range(1, self.n):
                j = self._l_subtraction(self.left_operand.range_array[i], z, self.right_operand.range_array)
                l += (self.left_operand.lower_array[i]-self.left_operand.upper_array[i-1]) * (1 - self.right_operand.lower_array[j])
                j = self._u_subtraction(self.left_operand.range_array[i-1], z, self.right_operand.range_array)
                u += (self.left_operand.upper_array[i] - self.left_operand.lower_array[i-1]) * (1 - self.right_operand.upper_array[j])
            uzk.append(min(u, 1))
            lzk.append(max(l, 0))
        self.output = ApproximatingPair(zk, uzk, lzk)

    def _u_subtraction(self, x, z, y_array):
        i = self.n - 1
        while i >= 0 and x - z < y_array[i]:
            i = i - 1
        if i >= 0:
            return i
        else:
            return 0

    def _l_subtraction(self, x, z, y_array):
        i = 0
        while i < self.n and y_array[i] < x - z:
            i = i + 1
        if i < self.n:
            return i
        else:
            return self.n - 1
