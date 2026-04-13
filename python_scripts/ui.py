def graphDim():
    x_low = float(input("enter lower bound of x domain (example: -1)\n"))
    x_high = float(input("enter upper bound of x domain (example: 1)\n"))
    y_low = float(input("enter lower bound of y domain (example: -1)\n"))
    y_high = float(input("enter upper bound of y domain (example: 1)\n"))

    return [x_low, x_high, y_low, y_high]

def Interface():
    stop = 0
    d = {}
    count_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    while stop == 0:
        string1 = "Enter Potential Flow Vizualization: \n 1) uniform \n 2) doublet \n 3) sink \n 4) source \n 5) vortex \n 6) tornado \n 7) whirlpool \n 8) exit \n"
        input1 = int(input(string1))
        if input1 == 8: #Exit
            stop = 1
        else:
            if input1 == 1: #Uniform
                input2 = float(input("Enter Strength?:"))
                count_list[input1] += 1
                d['uniform'+str(count_list[input1])] = [input2]
            elif input1 == 2: # Doublet
                list_it = []
                float2 = float(input("Enter Strength:"))
                float3 = float(input("Enter X Value:"))
                float4 = float(input("Enter Y Value:"))
                list_it = [float2, float3, float4]
                count_list[input1] += 1
                d['doublet'+str(count_list[input1])] = list_it
            elif input1 == 3: # Sink
                list_it = []
                float2 = float(input("Enter Strength:"))
                float3 = float(input("Enter Length:"))
                float4 = float(input("Enter X Value:"))
                float5 = float(input("Enter Y Value:"))
                list_it = [float2, float3, float4, float5]
                count_list[input1] += 1
                d['sink'+str(count_list[input1])] = list_it
            elif input1 == 4: # Source
                list_it = []
                float2 = float(input("Enter Strength:"))
                float3 = float(input("Enter Length:"))
                float4 = float(input("Enter X Value:"))
                float5 = float(input("Enter Y Value:"))
                list_it = [float2, float3, float4, float5]
                count_list[input1] += 1
                d['source'+str(count_list[input1])] = list_it

            elif input1 == 5: # Vortex
                list_it = []
                float2 = float(input("Enter Strength:"))
                float3 = float(input("Enter X Value:"))
                float4 = float(input("Enter Y Value:"))
                list_it = [float2, float3, float4]
                count_list[input1] += 1
                d['vortex'+str(count_list[input1])] = list_it

            elif input1 == 6: # Tornado
                list_it = []
                float2 = float(input("Enter Strength:"))
                float3 = float(input("Enter Vorticity:"))
                float4 = float(input("Enter X Value:"))
                float5 = float(input("Enter Y Value:"))
                list_it = [float2, float3, float4, float5]
                count_list[input1] += 1
                d['tornado'+str(count_list[input1])] = list_it

            elif input1 == 7: # Whirlpool
                list_it = []
                float2 = float(input("Enter Strength:"))
                float3 = float(input("Enter Vorticity:"))
                float4 = float(input("Enter X Value:"))
                float5 = float(input("Enter Y Value:"))
                list_it = [float2, float3, float4, float5]
                count_list[input1] += 1
                d['whirlpool'+str(count_list[input1])] = list_it
            else:
                print("Invalid input, returning to top")
    return d
