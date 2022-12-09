import awswrangler as wr

def main():
	# Checking/Creating Glue Catalog Databases
	if "billytest" not in wr.catalog.databases().values:
		wr.catalog.create_database("billytest")

	# Chceking/Creating Glue table
	try:
		print(wr.catalog.table(database="billytest", table="students"))
	except Exception as e:
		# Creating a Parquet Table
		path = "s3://billytest/dataset"
		res = wr.s3.store_parquet_metadata(
			path=path,
			database="billytest",
			table="students",
			dataset=True,
			mode="overwrite",
		)
		print("Table has been created: {0}".format(res))

	# Cheching Table metadata
	medadata = wr.catalog.table(database="billytest", table="students")
	print("Table 'students' medata is as the following:")
	print(medadata)

	# Reading data
	print("Table 'students' data is as the following:")
	df = wr.athena.read_sql_query("SELECT city, name, score FROM students", database="billytest")
	print(df)

	# Cleaning Up the Database
	# for table in wr.catalog.get_tables(database="billytest"):
 # 	    wr.catalog.delete_table_if_exists(database="billytest", table=table["students"])

if __name__ == '__main__':
	main()