class VanderwaalsParam:
    def __init__(
            self,
            inter_epsilon: float,
            inter_sigma: float,
            intra_epsilon: float,
            intra_sigma: float):
        self.inter_epsilon = inter_epsilon
        self.inter_sigma = inter_sigma
        self.intra_epsilon = intra_epsilon
        self.intra_sigma = intra_sigma
    
    def __str__(self) -> str:
        return f"{self.inter_epsilon}, {self.inter_sigma}, {self.intra_epsilon}, {self.intra_sigma}"


class ParamParser:
    @staticmethod
    def parse(file_):
        result = {}
        for line in file_:
            if line.startswith("#"):
                continue

            if line.startswith("NONBonded "):
                (
                    _,
                    type_,
                    inter_epsilon,
                    inter_sigma,
                    intra_epsilon,
                    intra_sigma,
                ) = line.split()

                result[type_] = VanderwaalsParam(
                    float(inter_epsilon),
                    float(inter_sigma),
                    float(intra_epsilon),
                    float(intra_sigma),
                )
            elif len(line.strip()) == 0:
                continue
            else:
                raise ValueError(f"Unparsable param line: {line}")

        return result