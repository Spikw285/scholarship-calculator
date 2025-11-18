from calculations import input_score, calculate_composite_score, calculate_required_regfinal
from config import DEFAULT_WEIGHTS


class Subject:
    def __init__(self):
        self.name = input("Enter subject name: ")
        self.teacher = input("Enter teacher's name: ")

        hours_input = input("Enter total hours: ")
        self.hours = float(hours_input) if hours_input else 0.0

        credits_input = input("Enter number of credits: ")
        self.credits = float(credits_input) if credits_input else 0.0

        while True:
            self.academic_period_type = input("Enter academic period type (semester/trimester): ").strip().lower()
            if self.academic_period_type in ["semester", "trimester"]:
                break
            else:
                print("Invalid academic period type. Please enter 'semester' or 'trimester'.")

        while True:
            target_input = input("Enter your target final score (70 for scholarship, 90 for increased scholarship): ")
            if not target_input:
                print("Target score cannot be empty.")
                continue
            try:
                self.target = float(target_input)
                if 0 <= self.target <= 100:
                    break
                else:
                    print("Target must be between 0 and 100.")
            except ValueError:
                print("Invalid input. Please enter a number.")

        self.weights = DEFAULT_WEIGHTS.get(self.academic_period_type, DEFAULT_WEIGHTS["trimester"])
        self.reg_mid = None
        self.reg_end = None
        self.reg_final = None
        self.reg_mid_breakdown = None
        self.reg_end_breakdown = None

    def input_scores(self):
        if self.academic_period_type == "trimester":
            self.reg_mid, self.reg_mid_breakdown = calculate_composite_score("RegMid (e.g., Midterm + Assignments)")
            self.reg_end, self.reg_end_breakdown = calculate_composite_score("RegEnd (e.g., Endterm + Assignments)")
        elif self.academic_period_type == "semester":
            self.reg_mid, self.reg_mid_breakdown = calculate_composite_score("Midterm (including any assignments)")
        self.reg_final = input_score("Enter your RegFinal (Final exam) score (or leave blank if unknown): ")

    def calculate_reg_term(self):
        if self.academic_period_type == "trimester":
            return (self.reg_mid + self.reg_end) / 2
        elif self.academic_period_type == "semester":
            return self.reg_mid
        else:
            return None

    def calculate_reg_total(self):
        reg_term = self.calculate_reg_term()
        if self.academic_period_type == "trimester":
            return reg_term * 0.6 + self.reg_final * 0.4
        elif self.academic_period_type == "semester":
            return self.reg_mid * (self.weights["reg_mid"] / 100) + self.reg_final * (self.weights["reg_final"] / 100)
        else:
            return None

    def display_progress(self):
        print(f"\nSubject: {self.name}")
        print(f"Teacher: {self.teacher}")
        print(f"Credits: {self.credits}, Hours: {self.hours}")
        if self.reg_mid is not None:
            print(f"RegMid: {self.reg_mid:.2f}")
            if self.reg_mid_breakdown:
                print("RegMid breakdown:")
                for i, comp in enumerate(self.reg_mid_breakdown, start=1):
                    if comp["type"] == "single":
                        print(f"  - Single value: {comp['score']:.2f}")
                    else:
                        print(
                            f"  - Component {i}: weight = {comp['weight']:.2f}%, score = {comp['score']:.2f}, contribution = {comp['contribution']:.2f}")
        if self.academic_period_type == "trimester" and self.reg_end is not None:
            print(f"RegEnd: {self.reg_end:.2f}")
            if self.reg_end_breakdown:
                print("RegEnd breakdown:")
                for i, comp in enumerate(self.reg_end_breakdown, start=1):
                    if comp["type"] == "single":
                        print(f"  - Single value: {comp['score']:.2f}")
                    else:
                        print(
                            f"  - Component {i}: weight = {comp['weight']:.2f}%, score = {comp['score']:.2f}, contribution = {comp['contribution']:.2f}")

        reg_term = self.calculate_reg_term()
        print(f"\nCalculated RegTerm: {reg_term:.2f}%")
        if self.reg_final is not None:
            print(f"RegFinal (Final exam score): {self.reg_final:.2f}")
            reg_total = self.calculate_reg_total()
            print(f"Calculated RegTotal: {reg_total:.2f}%")
            if reg_total >= self.target:
                print(f"🎉 Congratulations! You have achieved your target score of {self.target:.2f}% or higher. 🎉")
            else:
                required = calculate_required_regfinal(reg_term, self.target)
                effective_required = max(required, 50)
                print(
                    f"✅ To reach your target of {self.target:.2f}%, you need to score at least {effective_required:.2f}% on the Final Exam. ✅")
        else:
            print("RegFinal (Final exam score): Not provided")
            required = calculate_required_regfinal(reg_term, self.target)
            if required > 100:
                print(
                    f"⚠️ Unfortunately, with your current RegTerm score, it is impossible to reach your target of {self.target:.2f}% even with a perfect Final exam score. ⚠️")
            else:
                effective_required = max(required, 50)
                print(
                    f"✅ To reach your target of {self.target:.2f}%, you need to score at least {effective_required:.2f}% on the Final Exam. ✅")

        print("\nOverall Calculation Breakdown:")
        print("RegTotal = (RegTerm × 60%) + (RegFinal × 40%)")
        print(f"→ RegTerm × 60% = {reg_term:.2f} × 0.6 = {reg_term * 0.6:.2f}")
        if self.reg_final is not None:
            print(f"→ RegFinal × 40% = {self.reg_final:.2f} × 0.4 = {self.reg_final * 0.4:.2f}")
            print(
                f"→ RegTotal = ({reg_term:.2f} × 60%) + ({self.reg_final:.2f} × 40%) = {self.calculate_reg_total():.2f}")
        else:
            print("→ RegFinal × 40%: Not provided")
