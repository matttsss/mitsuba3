with open("times.txt") as file:
    lines = file.readlines()[:-1]
    times = sorted([int(line.split(" ")[-1].removesuffix("\n")) for line in lines])
    print("Time indexes: ", times)
    for i in range(len(times) - 1):
        if times[i] + 1 != times[i+1]:
            print(f"{times[i] = } and {times[i+1] = } ")
