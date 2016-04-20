'''
Databrics Cloud (Spark Cluster) File System is built upon HDFS, 
therefore, the commands to operate those files are HDFS commands
'''

# find all the support commands
dbutils.fs.help() 

# remove a folder
dbutils.fs.rm("/FileStore/tables", True)

# remove a file
dbutils.fs.rm("/FileStore/tables", False)

# list all the folders/files
display(dbutils.fs.ls("/FileStore/"))
