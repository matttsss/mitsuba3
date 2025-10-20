import drjit as dr
import mitsuba as mi

mi.set_variant("cuda_ad_rgb")

from helpers import generate_average

res = (512, 1024)

if __name__ == "__main__":
    monthly_average = mi.load_dict({
        "type": "timed_sunsky",
        "end_year": 2025,
        "end_month": 2,
        "end_day": 1
    })

    bimonthly_average = mi.load_dict({
        "type": "timed_sunsky",
        "end_year": 2025,
        "end_month": 3,
        "end_day": 1
    })

    samples_per_day = 500

    # TODO sort nb samples
    january_image = generate_average(monthly_average, res, 31*samples_per_day)

    # Update emitter to average in february
    monthly_params = mi.traverse(monthly_average)
    monthly_params["start_month"] = 2
    monthly_params["end_month"] = 3
    monthly_params.update()

    february_image = generate_average(monthly_average, res, 28*samples_per_day)

    average_of_average = (31 / 59) * january_image + (28 / 59) * february_image
    real_average = generate_average(bimonthly_average, res, (31 + 28)*samples_per_day)

    err = dr.mean(dr.abs(average_of_average - real_average) / (dr.abs(real_average) + 0.001), axis=None)

    assert err < 0.01, f"Average of average is incorrect {err = }"
