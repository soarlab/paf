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
        self.b = self.range_array[len(self.range_array) - 1]
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
        for i in range(0, n + 1):
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
        # The error_bound is only used for divisions by a RV whose range includes 0
        self.error_bound = 0.0

    def perform_operation(self):
        if self.operation == "+":
            if isinstance(self.left_operand, ApproximatingPair) and isinstance(self.right_operand, ApproximatingPair):
                self._perform_AP_Addition()
        if self.operation == "-":
            if isinstance(self.left_operand, ApproximatingPair) and isinstance(self.right_operand, ApproximatingPair):
                self._perform_AP_Subtraction()
        if self.operation == "*":
            if isinstance(self.left_operand, ApproximatingPair) and isinstance(self.right_operand, ApproximatingPair):
                self._perform_AP_Multiplication()
        if self.operation == "/":
            if isinstance(self.left_operand, ApproximatingPair) and isinstance(self.right_operand, ApproximatingPair):
                self._perform_AP_Division()
        else:
            raise ValueError("Operation must be +, - , * or /")


    def _perform_AP_Addition(self):
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
                l += (self.left_operand.lower_array[i] - self.left_operand.upper_array[i - 1]) * \
                     self.right_operand.lower_array[j]
                j = self._u_addition(self.left_operand.range_array[i - 1], z, self.right_operand.range_array)
                u += (self.left_operand.upper_array[i] - self.left_operand.lower_array[i - 1]) * \
                     self.right_operand.upper_array[j]
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

    def _perform_AP_Subtraction(self):
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
                l += (self.left_operand.lower_array[i] - self.left_operand.upper_array[i - 1]) * (
                        1 - self.right_operand.upper_array[j])
                j = self._u_subtraction(self.left_operand.range_array[i - 1], z, self.right_operand.range_array)
                u += (self.left_operand.upper_array[i] - self.left_operand.lower_array[i - 1]) * (
                        1 - self.right_operand.lower_array[j])
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

    # {PRECONDITION: if 0 is in the range of left_operand then 0 must be a point of discontinuity of left_operand}
    def _perform_AP_Multiplication(self):
        ax = self.left_operand.a
        bx = self.left_operand.b
        ay = self.right_operand.a
        by = self.right_operand.b
        # Compute range using interval arithmetic
        a = min(ax * ay, ax * by, bx * ay, bx * by)
        b = max(ax * ay, ax * by, bx * ay, bx * by)
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
            if z >= 0:
                for i in range(1, self.n):
                    if 0 <= self.left_operand.range_array[i - 1]:
                        j = self._l_multiplication(self.left_operand.range_array[i], z,
                                                   self.right_operand.range_array)
                        l += (self.left_operand.lower_array[i] - self.left_operand.upper_array[i - 1]) * \
                             self.right_operand.lower_array[j]
                        j = self._u_multiplication(self.left_operand.range_array[i - 1], z,
                                                   self.right_operand.range_array)
                        u += (self.left_operand.upper_array[i] - self.left_operand.lower_array[i - 1]) * \
                             self.right_operand.upper_array[j]
                    elif self.left_operand.range_array[i] < 0:
                        j = self._l_multiplication(self.left_operand.range_array[i - 1], z,
                                                   self.right_operand.range_array)
                        l += (self.left_operand.lower_array[i] - self.left_operand.upper_array[i - 1]) * \
                             (1 - self.right_operand.upper_array[j])
                        j = self._u_multiplication(self.left_operand.range_array[i], z,
                                                   self.right_operand.range_array)
                        u += (self.left_operand.upper_array[i] - self.left_operand.lower_array[i - 1]) * \
                             (1 - self.right_operand.lower_array[j])
                    else:
                        raise ValueError("0 must be a discontinuity point")
            else:
                for i in range(1, self.n):
                    if 0 <= self.left_operand.range_array[i - 1]:
                        j = self._l_multiplication(self.left_operand.range_array[i], z,
                                                   self.right_operand.range_array)
                        l += (self.left_operand.lower_array[i] - self.left_operand.upper_array[i - 1]) * \
                             self.right_operand.upper_array[j]
                        j = self._u_multiplication(self.left_operand.range_array[i - 1], z,
                                                   self.right_operand.range_array)
                        u += (self.left_operand.upper_array[i] - self.left_operand.lower_array[i - 1]) * \
                             self.right_operand.upper_array[j]
                    elif self.left_operand.range_array[i] < 0:
                        j = self._l_multiplication(self.left_operand.range_array[i], z,
                                                   self.right_operand.range_array)
                        l += (self.left_operand.lower_array[i] - self.left_operand.upper_array[i - 1]) * \
                             (1 - self.right_operand.upper_array[j])
                        j = self._u_multiplication(self.left_operand.range_array[i - 1], z,
                                                   self.right_operand.range_array)
                        u += (self.left_operand.upper_array[i] - self.left_operand.lower_array[i - 1]) * \
                             (1 - self.right_operand.lower_array[j])
                    else:
                        raise ValueError("0 must be a discontinuity point")
            uzk.append(min(u, 1))
            lzk.append(max(l, 0))
        self.output = ApproximatingPair(zk, uzk, lzk)

    def _u_multiplication(self, x, z, y_array):
        if x >= 0:
            i = self.n - 1
            while i >= 0 and z / x <= y_array[i]:
                i = i - 1
            if i >= 0:
                return i
            else:
                return 0
        else:
            i = 0
            while i < self.n and y_array[i] < z / x:
                i = i + 1
            if i < self.n:
                return i
            else:
                return self.n - 1

    def _l_multiplication(self, x, z, y_array):
        if x >= 0:
            i = 0
            while i < self.n and y_array[i] < z / x:
                i = i + 1
            if i < self.n:
                return i
            else:
                return self.n - 1
        else:
            i = self.n - 1
            while i >= 0 and z / x <= y_array[i]:
                i = i - 1
            if i >= 0:
                return i
            else:
                return 0

    # {INFORMAL PRECONDITION: if the range of the right operand Y contains 0, then there must exist discontinuity points
    # u,v just below and just above 0 such that the probability that Y lies between u and v is small.
    # This is because the routine is going to remove the mass in this interval}
    def _perform_AP_Division(self):
        ax = self.left_operand.a
        bx = self.left_operand.b
        ay = self.right_operand.a
        by = self.right_operand.b
        # Compute the upper or lower cdf of the right operand at zero and the range of values
        if ay < 0 < by:
            i = 0
            while self.right_operand.range_array[i] < 0:
                i += 1
            y_plus_0 = self.right_operand.upper_array[i - 1]
            y_minus_0 = self.right_operand.lower_array[i - 1]
            u = self.right_operand.range_array[i - 1]
            v = self.right_operand.range_array[i]
            a = min(ax / u, ax / v, bx / u, bx / v)
            b = max(ax / u, ax / v, bx / u, bx / v)
            self.error_bound = self.right_operand.upper_array[i] - self.right_operand.lower_array[i]
        else:
            a = min(ax / ay, ax / by, bx / ay, bx / by)
            b = max(ax / ay, ax / by, bx / ay, bx / by)
            if 0 < ay:
                y_plus_0 = 0
                y_minus_0 = 0
            elif by < 1:
                y_plus_0 = 1
                y_minus_0 = 1
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
            if z > 0:
                for i in range(1, self.n):
                    if 0 < self.left_operand.range_array[i - 1]:
                        j = self._l_division(self.left_operand.range_array[i], z,
                                             self.right_operand.range_array)
                        l += (self.left_operand.lower_array[i] - self.left_operand.upper_array[i - 1]) * \
                             ((1 - self.right_operand.upper_array[j]) + y_minus_0)
                        j = self._u_division(self.left_operand.range_array[i - 1], z,
                                             self.right_operand.range_array)
                        u += (self.left_operand.upper_array[i] - self.left_operand.lower_array[i - 1]) * \
                             ((1 - self.right_operand.lower_array[j]) + y_plus_0)
                    elif self.left_operand.range_array[i] < 0:
                        j = self._l_division(self.left_operand.range_array[i - 1], z,
                                             self.right_operand.range_array)
                        l += (self.left_operand.lower_array[i] - self.left_operand.upper_array[i - 1]) * \
                             (self.right_operand.lower_array[j] + (1 - y_plus_0))
                        j = self._u_division(self.left_operand.range_array[i], z,
                                             self.right_operand.range_array)
                        u += (self.left_operand.upper_array[i] - self.left_operand.lower_array[i - 1]) * \
                             (self.right_operand.upper_array[j] + (1 - y_minus_0))
            else:
                for i in range(1, self.n):
                    if 0 < self.left_operand.range_array[i - 1]:
                        j = self._l_division(self.left_operand.range_array[i - 1], z,
                                             self.right_operand.range_array)
                        l += (self.left_operand.lower_array[i] - self.left_operand.upper_array[i - 1]) * \
                             (1 - self.right_operand.upper_array[j])
                        j = self._u_division(self.left_operand.range_array[i], z,
                                             self.right_operand.range_array)
                        u += (self.left_operand.upper_array[i] - self.left_operand.lower_array[i - 1]) * \
                             (1 - self.right_operand.lower_array[j])
                    elif self.left_operand.range_array[i] < 0:
                        j = self._l_division(self.left_operand.range_array[i], z,
                                             self.right_operand.range_array)
                        l += (self.left_operand.lower_array[i] - self.left_operand.upper_array[i - 1]) * \
                            self.right_operand.lower_array[j]
                        j = self._u_division(self.left_operand.range_array[i - 1], z,
                                             self.right_operand.range_array)
                        u += (self.left_operand.upper_array[i] - self.left_operand.lower_array[i - 1]) * \
                            self.right_operand.upper_array[j]
            uzk.append(min(u, 1))
            lzk.append(max(l, 0))
        self.output = ApproximatingPair(zk, uzk, lzk)

    def _u_division(self, x, z, y_array):
        if x < 0:
            i = self.n - 1
            while i >= 0 and x / z <= y_array[i]:
                i = i - 1
            if i >= 0:
                return i
            else:
                return 0
        else:
            i = 0
            while i < self.n and y_array[i] < x / z:
                i = i + 1
            if i < self.n:
                return i
            else:
                return self.n - 1

    def _l_division(self, x, z, y_array):
        if x < 0:
            i = 0
            while i < self.n and y_array[i] < x / z:
                i = i + 1
            if i < self.n:
                return i
            else:
                return self.n - 1
        else:
            i = self.n - 1
            while i >= 0 and x / z <= y_array[i]:
                i = i - 1
            if i >= 0:
                return i
            else:
                return 0
