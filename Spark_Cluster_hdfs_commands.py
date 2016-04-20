'''
Databrics Cloud (Spark Cluster) File System is built upon HDFS, 
therefore, the commands to operate those files are HDFS commands
'''

# check all the support commands in a cell
dbutils.fs.help() 

# remove a folder
rm("/FileStore/tables", True)

# remove a file
rm("/FileStore/tables", False)

# list all the folders/files
display(dbutils.fs.ls("/FileStore/"))
