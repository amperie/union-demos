import union


image = union.ImageSpec(
    builder="envd",
    base_image="ghcr.io/unionai-oss/union:py3.10-latest",
    name="data",
    registry="pablounionai",
    packages=["union"],
)


# Task to calculate monthly interest payment on a loan
@union.task(
    container_image=image
)
def calculate_interest(principal: int, rate: float, time: int) -> float:
    print(f"principal: {principal}, rate: {rate}, time: {time}")
    return (principal * rate * time) / 12


# Workflow using the calculate_interest task
@union.workflow
def interest_workflow(principal: int, rate: float, time: int) -> float:
    return calculate_interest(principal=principal, rate=rate, time=time)


# Create LaunchPlan for interest_workflow
lp = union.LaunchPlan.get_or_create(
    workflow=interest_workflow,
    name="interest_workflow_lp",
)


# Mapping over the launch plan to calculate interest for multiple loans
@union.workflow
def map_interest_wf() -> list[float]:
    principal = [1000, 5000, 10000]
    rate = [0.05, 0.04, 0.03]  # Different interest rates for each loan
    time = [12, 24, 36]        # Loan periods in months
    return union.map(lp)(principal=principal, rate=rate, time=time)


# Mapping over the launch plan to calculate interest for
# multiple loans while fixing an input
@union.workflow
def map_interest_fixed_principal_wf() -> list[float]:
    rate = [0.05, 0.04, 0.03]  # Different interest rates for each loan
    time = [12, 24, 36]        # Loan periods in months
    # Note: principal is set to 1000 for all the calculations
    return union.map(
        lp, bound_inputs={'principal': 1000})(rate=rate, time=time)
