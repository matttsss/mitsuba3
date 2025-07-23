import drjit as dr
import mitsuba as mi

mi.set_variant("cuda_ad_rgb")

from helpers import render

res = (512, 1024)

if __name__ == "__main__":
    monthly_average = mi.load_dict({
        "type": "avg_sunsky",
        "time_resolution": 20,
        "end_year": 2025,
        "end_month": 2,
        "end_day": 1
    })

    bimonthly_average = mi.load_dict({
        "type": "avg_sunsky",
        "time_resolution": 20,
        "end_year": 2025,
        "end_month": 3,
        "end_day": 1
    })

    january_image = render(monthly_average, res)

    monthly_params = mi.traverse(monthly_average)
    monthly_params["start_month"] = 2
    monthly_params["end_month"] = 3
    monthly_params.update()

    february_image = render(monthly_average, res)

    average_of_average = (31 / 59) * january_image + (28 / 59) * february_image
    real_average = render(bimonthly_average, res)


    january_image = mi.Bitmap(january_image).convert(mi.Bitmap.PixelFormat.RGB, mi.Bitmap.Float32) 
    february_image = mi.Bitmap(february_image).convert(mi.Bitmap.PixelFormat.RGB, mi.Bitmap.Float32) 
    average_of_average = mi.Bitmap(average_of_average).convert(mi.Bitmap.PixelFormat.RGB, mi.Bitmap.Float32)
    real_average = mi.Bitmap(real_average).convert(mi.Bitmap.PixelFormat.RGB, mi.Bitmap.Float32)
    
    

    mi.util.write_bitmap("results/average_of_january.exr", january_image)
    mi.util.write_bitmap("results/average_of_february.exr", february_image)
    mi.util.write_bitmap("results/average_of_average.exr", average_of_average)
    mi.util.write_bitmap("results/real_average.exr", real_average)
