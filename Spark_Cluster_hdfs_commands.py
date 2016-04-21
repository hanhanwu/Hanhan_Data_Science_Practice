'''
Databrics Cloud (Spark Cluster) File System is built upon HDFS, 
therefore, the commands to operate those files are HDFS commands
'''

# find all the support commands
dbutils.fs.help() 

# remove a folder
dbutils.fs.rm("/FileStore/tables", True)
# or
%fs rm -r "/FileStore/tables"

# remove a file
dbutils.fs.rm("/FileStore/tables/f1", False)
# or
%fs rm  "/FileStore/tables/f1"

# list all the folders/files
display(dbutils.fs.ls("/FileStore/"))
# or
%fs ls "/FileStore/"
