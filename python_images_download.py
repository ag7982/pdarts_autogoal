from google_images_download import google_images_download 
  
# creating object
response = google_images_download.googleimagesdownload() 
  
def downloadimages(query, count):
    # keywords is the search query
    # format is the image file format
    # limit is the number of images to be downloaded
    # print urs is to print the image file url
    # size is the image size which can
    # be specified manually ("large, medium, icon")
    # aspect ratio denotes the height width ratio
    # of images to download. ("tall, square, wide, panoramic")
    arguments = {"keywords": query,
                 "limit":count,
                 "print_urls":True,
                 "format":"jpg",
                 "size": "medium"
                 }
  
    paths=response.download(arguments)
      
    #print(paths)
  
# Driver Code

query=input("Enter image category to download: ")
count=input("Enter number of images you want (limit is 100):" )
downloadimages(query, int(count)) 
print()
    
    
    
    
