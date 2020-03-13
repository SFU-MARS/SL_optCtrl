import pandas as pd
import numpy as np


def trunc(df, tra_len):
	df_trunc = pd.DataFrame(columns = df.columns)
	stop = False
	count = 0
	trash = 0
	non_trash = 0
	inf = 0
	ban_list = {np.inf, -np.inf}

	print("Total rows: ", len(df))

	for index, row in df.iterrows():
		count += 1
		if (stop):
			if (count == tra_len):
				count = 0
				stop = False
			trash += 1
			continue

		df_trunc = df_trunc.append(row)
		non_trash += 1
		if (row['value'] in {1000, -400}):
			stop = True

	print("rows trunc: ", trash)

	df_final = pd.DataFrame(columns = df_trunc.columns)

	for index, row in df_trunc.iterrows():
		if (row['d1'] in ban_list  or
			row['d2'] in ban_list  or
			row['d3'] in ban_list  or
			row['d4'] in ban_list  or
			row['d5'] in ban_list  or
			row['d6'] in ban_list  or
			row['d7'] in ban_list  or
			row['d8'] in ban_list):
			inf += 1
		else:
			df_final = df_final.append(row)

	print("inf rows: ", inf)
	print("current rows: ", len(df_final))
	df_final.to_csv("feasible_trunc_2.csv")


def special_func_combine(file_list):
    df1 = pd.read_csv(file_list[0])
    df2 = pd.read_csv(file_list[1])
    final_df = pd.concat([df1, df2], sort=False)
    for i in range(2, len(file_list)):
    	df = pd.read_csv(file_list[i])
    	final_df = pd.concat([final_df, df], sort=False)

    print(len(final_df))
    final_df.to_csv("final_data.csv")



# file = "./valFunc_mpc_filled_2.csv"
# df = pd.read_csv(file)
# print(df.dtypes)
# feas = df[df['col_trajectory_flag'] == 2]
# infeas = df[df['col_trajectory_flag'] == 3]

# trunc(feas, 80)
# trunc(infeas, 140)
# print(len(feas) / 140)
# print(len(infeas) / 140)
file_list = ["./infeasible_trunc_0.csv", "./infeasible_trunc_1.csv", "./feasible_trunc_2.csv"]
special_func_combine(file_list)
