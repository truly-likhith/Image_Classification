

```python
!pip install bing-image-downloader
```

    Requirement already satisfied: bing-image-downloader in f:\anaconda\lib\site-packages (1.0.4)
    


```python
!pip install ipython-autotime
%load_ext autotime
```

    Requirement already satisfied: ipython-autotime in f:\anaconda\lib\site-packages (0.2.0)
    Requirement already satisfied: ipython in f:\anaconda\lib\site-packages (from ipython-autotime) (7.4.0)
    Requirement already satisfied: decorator in f:\anaconda\lib\site-packages (from ipython->ipython-autotime) (4.4.0)
    Requirement already satisfied: colorama; sys_platform == "win32" in f:\anaconda\lib\site-packages (from ipython->ipython-autotime) (0.4.1)
    Requirement already satisfied: pygments in f:\anaconda\lib\site-packages (from ipython->ipython-autotime) (2.3.1)
    Requirement already satisfied: jedi>=0.10 in f:\anaconda\lib\site-packages (from ipython->ipython-autotime) (0.13.3)
    Requirement already satisfied: prompt-toolkit<2.1.0,>=2.0.0 in f:\anaconda\lib\site-packages (from ipython->ipython-autotime) (2.0.9)
    Requirement already satisfied: traitlets>=4.2 in f:\anaconda\lib\site-packages (from ipython->ipython-autotime) (4.3.2)
    Requirement already satisfied: backcall in f:\anaconda\lib\site-packages (from ipython->ipython-autotime) (0.1.0)
    Requirement already satisfied: pickleshare in f:\anaconda\lib\site-packages (from ipython->ipython-autotime) (0.7.5)
    Requirement already satisfied: setuptools>=18.5 in f:\anaconda\lib\site-packages (from ipython->ipython-autotime) (50.3.2)
    Requirement already satisfied: parso>=0.3.0 in f:\anaconda\lib\site-packages (from jedi>=0.10->ipython->ipython-autotime) (0.3.4)
    Requirement already satisfied: six>=1.9.0 in f:\anaconda\lib\site-packages (from prompt-toolkit<2.1.0,>=2.0.0->ipython->ipython-autotime) (1.12.0)
    Requirement already satisfied: wcwidth in f:\anaconda\lib\site-packages (from prompt-toolkit<2.1.0,>=2.0.0->ipython->ipython-autotime) (0.1.7)
    Requirement already satisfied: ipython-genutils in f:\anaconda\lib\site-packages (from traitlets>=4.2->ipython->ipython-autotime) (0.2.0)
    The autotime extension is already loaded. To reload it, use:
      %reload_ext autotime
    time: 18.3 s
    


```python
!mkdir images
```

    time: 15 ms
    

### LOADING THE IMAGES DIRECT FROM THE NET
- Images being scrapped from BING (library)


```python
#limit acutally states that no of images we need

from  bing_image_downloader import downloader

# downloaded these all and stored in "OUTPUT_DIR"
# downloader.download("pretty sunflower",limit=20,output_dir='image_classyy',adult_filter_off=True)
# downloader.download("rugby ball leather",limit=20,output_dir='image_classyy',adult_filter_off=True)4

downloader.download("ice cream cone",limit=20,output_dir='image_classyy',adult_filter_off=True)
```

    
    
    [!!]Indexing page: 1
    
    [%] Indexed 20 Images on Page 1.
    
    ===============================================
    
    [%] Downloading Image #1 from http://www.camdenac.com/wp-content/uploads/2017/05/7-scoop-ice-cream-cone.jpg
    [%] File Downloaded !
    
    [%] Downloading Image #2 from https://shewearsmanyhats.com/wp-content/uploads/2013/07/dipped-ice-cream-cones-7.jpg
    [%] File Downloaded !
    
    [%] Downloading Image #3 from https://www.webstaurantstore.com/images/products/extra_large/67189/1915151.jpg
    [%] File Downloaded !
    
    [%] Downloading Image #4 from https://www.procaffenation.com/wp-content/uploads/2017/04/21-compressor-1.jpg
    [%] File Downloaded !
    
    [%] Downloading Image #5 from https://sugarspunrun.com/wp-content/uploads/2018/07/Ice-Cream-Cone-Cupcakes-Recipe-1-of-1-6.jpg
    [%] File Downloaded !
    
    [%] Downloading Image #6 from https://upload.wikimedia.org/wikipedia/commons/d/da/Strawberry_ice_cream_cone_%285076899310%29.jpg
    [%] File Downloaded !
    
    [%] Downloading Image #7 from https://reneenicoleskitchen.com/wp-content/uploads/2017/05/Dipped-Ice-Cream-Cones-Image-2.jpg
    [%] File Downloaded !
    
    [%] Downloading Image #8 from https://www.cravingsofalunatic.com/wp-content/uploads/2016/03/Ice-Cream-Cone-Cupcakes-3.jpg
    [%] File Downloaded !
    
    [%] Downloading Image #9 from http://graphics8.nytimes.com/images/2013/06/02/magazine/02wmt/02wmt-superJumbo-v3.jpg
    [%] File Downloaded !
    
    [%] Downloading Image #10 from https://americacomesalive.com/i/ice-cream-cone.jpg
    [!] Issue getting: https://americacomesalive.com/i/ice-cream-cone.jpg
    [!] Error:: HTTP Error 403: Forbidden
    [%] Downloading Image #10 from https://www.webstaurantstore.com/images/products/extra_large/123922/944570.jpg
    [%] File Downloaded !
    
    [%] Downloading Image #11 from https://gabrielquotes.files.wordpress.com/2012/04/ice-cream-cone-4650355-ji.jpg
    [!] Issue getting: https://gabrielquotes.files.wordpress.com/2012/04/ice-cream-cone-4650355-ji.jpg
    [!] Error:: [WinError 10054] An existing connection was forcibly closed by the remote host
    [%] Downloading Image #11 from https://pixfeeds.com/images/desserts/ice-creams/1280-177118229-ice-cream-cone.jpg
    [!] Issue getting: https://pixfeeds.com/images/desserts/ice-creams/1280-177118229-ice-cream-cone.jpg
    [!] Error:: <urlopen error [Errno 11001] getaddrinfo failed>
    [%] Downloading Image #11 from http://4.bp.blogspot.com/-NWgPAda_Mas/VObrOYGy7qI/AAAAAAAAptQ/1u5ijmftcpw/s1600/P1530546.JPG
    [!] Issue getting: http://4.bp.blogspot.com/-NWgPAda_Mas/VObrOYGy7qI/AAAAAAAAptQ/1u5ijmftcpw/s1600/P1530546.JPG
    [!] Error:: <urlopen error [Errno 11001] getaddrinfo failed>
    [%] Downloading Image #11 from http://www.ellinorsicecream.co.uk/wp-content/uploads/scoop-for-trailor-2.jpg
    [!] Issue getting: http://www.ellinorsicecream.co.uk/wp-content/uploads/scoop-for-trailor-2.jpg
    [!] Error:: <urlopen error [Errno 11001] getaddrinfo failed>
    [%] Downloading Image #11 from http://momvstheboys.com/wp-content/uploads/2013/06/ice-cream-cone-cupcakes.jpg
    [!] Issue getting: http://momvstheboys.com/wp-content/uploads/2013/06/ice-cream-cone-cupcakes.jpg
    [!] Error:: <urlopen error [Errno 11001] getaddrinfo failed>
    [%] Downloading Image #11 from https://pixfeeds.com/images/desserts/ice-creams/1280-155381237-ice-cream-cone-with-walnuts.jpg
    [!] Issue getting: https://pixfeeds.com/images/desserts/ice-creams/1280-155381237-ice-cream-cone-with-walnuts.jpg
    [!] Error:: <urlopen error [Errno 11001] getaddrinfo failed>
    [%] Downloading Image #11 from https://cdn-image.foodandwine.com/sites/default/files/1497371785/ice-cream-cones-history-FT-BLOG0617.jpg
    [!] Issue getting: https://cdn-image.foodandwine.com/sites/default/files/1497371785/ice-cream-cones-history-FT-BLOG0617.jpg
    [!] Error:: <urlopen error [Errno 11001] getaddrinfo failed>
    [%] Downloading Image #11 from http://cdnimg.webstaurantstore.com/images/products/extra_large/67189/507430.jpg
    [!] Issue getting: http://cdnimg.webstaurantstore.com/images/products/extra_large/67189/507430.jpg
    [!] Error:: <urlopen error [Errno 11001] getaddrinfo failed>
    [%] Downloading Image #11 from http://www.cravingsofalunatic.com/wp-content/uploads/2016/03/Ice-Cream-Cone-Cupcakes-4.jpg
    [!] Issue getting: http://www.cravingsofalunatic.com/wp-content/uploads/2016/03/Ice-Cream-Cone-Cupcakes-4.jpg
    [!] Error:: <urlopen error [Errno 11001] getaddrinfo failed>
    
    
    [!!]Indexing page: 2
    
    


    ---------------------------------------------------------------------------

    gaierror                                  Traceback (most recent call last)

    F:\anaconda\lib\urllib\request.py in do_open(self, http_class, req, **http_conn_args)
       1316                 h.request(req.get_method(), req.selector, req.data, headers,
    -> 1317                           encode_chunked=req.has_header('Transfer-encoding'))
       1318             except OSError as err: # timeout error
    

    F:\anaconda\lib\http\client.py in request(self, method, url, body, headers, encode_chunked)
       1228         """Send a complete request to the server."""
    -> 1229         self._send_request(method, url, body, headers, encode_chunked)
       1230 
    

    F:\anaconda\lib\http\client.py in _send_request(self, method, url, body, headers, encode_chunked)
       1274             body = _encode(body, 'body')
    -> 1275         self.endheaders(body, encode_chunked=encode_chunked)
       1276 
    

    F:\anaconda\lib\http\client.py in endheaders(self, message_body, encode_chunked)
       1223             raise CannotSendHeader()
    -> 1224         self._send_output(message_body, encode_chunked=encode_chunked)
       1225 
    

    F:\anaconda\lib\http\client.py in _send_output(self, message_body, encode_chunked)
       1015         del self._buffer[:]
    -> 1016         self.send(msg)
       1017 
    

    F:\anaconda\lib\http\client.py in send(self, data)
        955             if self.auto_open:
    --> 956                 self.connect()
        957             else:
    

    F:\anaconda\lib\http\client.py in connect(self)
       1383 
    -> 1384             super().connect()
       1385 
    

    F:\anaconda\lib\http\client.py in connect(self)
        927         self.sock = self._create_connection(
    --> 928             (self.host,self.port), self.timeout, self.source_address)
        929         self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    

    F:\anaconda\lib\socket.py in create_connection(address, timeout, source_address)
        706     err = None
    --> 707     for res in getaddrinfo(host, port, 0, SOCK_STREAM):
        708         af, socktype, proto, canonname, sa = res
    

    F:\anaconda\lib\socket.py in getaddrinfo(host, port, family, type, proto, flags)
        747     addrlist = []
    --> 748     for res in _socket.getaddrinfo(host, port, family, type, proto, flags):
        749         af, socktype, proto, canonname, sa = res
    

    gaierror: [Errno 11001] getaddrinfo failed

    
    During handling of the above exception, another exception occurred:
    

    URLError                                  Traceback (most recent call last)

    <ipython-input-5-11eb4b37eb5d> in <module>
          1 #limit acutally states that no of images we need
          2 from  bing_image_downloader import downloader
    ----> 3 downloader.download("ice cream cone",limit=20,output_dir='image_classyy',adult_filter_off=True)
    

    F:\anaconda\lib\site-packages\bing_image_downloader\downloader.py in download(query, limit, output_dir, adult_filter_off, force_replace, timeout)
         33 
         34     bing = Bing(query, limit, output_dir, adult, timeout)
    ---> 35     bing.run()
         36 
         37 
    

    F:\anaconda\lib\site-packages\bing_image_downloader\bing.py in run(self)
         68                           + '&adlt=' + self.adult + '&qft=' + self.filters
         69             request = urllib.request.Request(request_url, None, headers=self.headers)
    ---> 70             response = urllib.request.urlopen(request)
         71             html = response.read().decode('utf8')
         72             links = re.findall('murl&quot;:&quot;(.*?)&quot;', html)
    

    F:\anaconda\lib\urllib\request.py in urlopen(url, data, timeout, cafile, capath, cadefault, context)
        220     else:
        221         opener = _opener
    --> 222     return opener.open(url, data, timeout)
        223 
        224 def install_opener(opener):
    

    F:\anaconda\lib\urllib\request.py in open(self, fullurl, data, timeout)
        523             req = meth(req)
        524 
    --> 525         response = self._open(req, data)
        526 
        527         # post-process response
    

    F:\anaconda\lib\urllib\request.py in _open(self, req, data)
        541         protocol = req.type
        542         result = self._call_chain(self.handle_open, protocol, protocol +
    --> 543                                   '_open', req)
        544         if result:
        545             return result
    

    F:\anaconda\lib\urllib\request.py in _call_chain(self, chain, kind, meth_name, *args)
        501         for handler in handlers:
        502             func = getattr(handler, meth_name)
    --> 503             result = func(*args)
        504             if result is not None:
        505                 return result
    

    F:\anaconda\lib\urllib\request.py in https_open(self, req)
       1358         def https_open(self, req):
       1359             return self.do_open(http.client.HTTPSConnection, req,
    -> 1360                 context=self._context, check_hostname=self._check_hostname)
       1361 
       1362         https_request = AbstractHTTPHandler.do_request_
    

    F:\anaconda\lib\urllib\request.py in do_open(self, http_class, req, **http_conn_args)
       1317                           encode_chunked=req.has_header('Transfer-encoding'))
       1318             except OSError as err: # timeout error
    -> 1319                 raise URLError(err)
       1320             r = h.getresponse()
       1321         except:
    

    URLError: <urlopen error [Errno 11001] getaddrinfo failed>


    time: 2min 5s
    

### PREPROCESSING


```python
import os
import matplotlib.pyplot as plt
import numpy as np
```

    time: 0 ns
    


```python
from skimage.io import imread
```

    time: 0 ns
    


```python
from skimage.transform import resize
```

    time: 0 ns
    


```python
# Flatten is used to convert the matrix dimmensions
# EXAMPLE

a = np.array([[1,2,3],
              [4,5,6]])
print("Before Flatten: ",a)
print("After Flatten: ",a.flatten())
```

    Before Flatten:  [[1 2 3]
     [4 5 6]]
    After Flatten:  [1 2 3 4 5 6]
    time: 0 ns
    


```python
target = []
images = []
flat_data = []
DATADIR = 'image_classyy'
CATEGORIES = ['pretty sunflower','rugby ball leather','ice cream cone']
```

    time: 15 ms
    


```python
for categories in CATEGORIES:
    class_num = CATEGORIES.index(categories)
    print(class_num)
```

    0
    1
    2
    time: 0 ns
    


```python
# Let's print the path of each categories

for categories in CATEGORIES:
    class_num = CATEGORIES.index(categories) # labels for the each categories (index-based)
    path = os.path.join(DATADIR,categories) # creating path to use all images
    print(class_num)
    print(path)
    
    for img in os.listdir(path):
        img_array = imread(os.path.join(path,img))
        #plt.imshow(img_array)
        plt.axis("off")
        img_resized = resize(img_array,(150,150,3))
        flat_data.append(img_resized.flatten())
        images.append(img_resized)
        target.append(class_num) # target will be our indexs
# converting to travel the entire as like as array

flat_data = np.array(flat_data) #converting into flat_data
target = np.array(target) 
images = np.array(images)
```

    0
    image_classyy\pretty sunflower
    1
    image_classyy\rugby ball leather
    2
    image_classyy\ice cream cone
    


![png](output_12_1.png)


    time: 7.61 s
    

# PRE PROCESSING
- 1) RESIZE
- 2) FLATTEN


```python
flat_data
```




    array([[0.03137255, 0.04313725, 0.        , ..., 0.01861351, 0.01469194,
            0.        ],
           [0.03137255, 0.04313725, 0.        , ..., 0.01861351, 0.01469194,
            0.        ],
           [0.11337255, 0.16435294, 0.32905882, ..., 0.07764706, 0.0854902 ,
            0.08156863],
           ...,
           [0.81176471, 0.83137255, 0.84705882, ..., 0.83137255, 0.81568627,
            0.81960784],
           [0.55721569, 0.41881089, 0.3302366 , ..., 0.33681046, 0.34473333,
            0.38130806],
           [1.        , 1.        , 1.        , ..., 1.        , 1.        ,
            1.        ]])



    time: 47 ms
    


```python
flat_data[0]
```




    array([0.03137255, 0.04313725, 0.        , ..., 0.01861351, 0.01469194,
           0.        ])



    time: 0 ns
    


```python
len(flat_data[0])
```




    67500



    time: 0 ns
    


```python
flat_data.shape
```




    (61, 67500)



    time: 16 ms
    


```python
target
```




    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])



    time: 0 ns
    


```python
target.shape
```




    (61,)



    time: 0 ns
    

### SPLITING THE DATA INTO TRAINING AND TESTING


```python
from sklearn.model_selection import train_test_split
```

    time: 8.92 s
    


```python
X_train,X_test,Y_train,Y_test = train_test_split(flat_data,target,test_size=0.3,random_state=42)
```

    time: 31 ms
    


```python
X_train.shape
```




    (42, 67500)



    time: 15 ms
    


```python
X_test.shape
```




    (19, 67500)



    time: 0 ns
    


```python
Y_train.shape
```




    (42,)



    time: 31 ms
    


```python
Y_test.shape
```




    (19,)



    time: 0 ns
    


```python
from sklearn.model_selection import GridSearchCV
from sklearn import svm
param_grid = [
    {'C':[1,10,100,1000],'kernel':['linear']},
    {'C':[1,10,100,1000],'gamma':[0.001,0.0001],'kernel':['rbf']},
]
# to understand along with this image probabloty and another image probablity line bar to see what % accuracte 
svc = svm.SVC(probability=True)
clf = GridSearchCV(svc,param_grid)
clf.fit(X_train,Y_train)
```

    F:\anaconda\lib\site-packages\sklearn\model_selection\_split.py:2053: FutureWarning: You should specify a value for 'cv' instead of relying on the default value. The default value will change from 3 to 5 in version 0.22.
      warnings.warn(CV_WARNING, FutureWarning)
    F:\anaconda\lib\site-packages\sklearn\model_selection\_search.py:841: DeprecationWarning: The default of the `iid` parameter will change from True to False in version 0.22 and will be removed in 0.24. This will change numeric results when test-set sizes are unequal.
      DeprecationWarning)
    




    GridSearchCV(cv='warn', error_score='raise-deprecating',
           estimator=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
      kernel='rbf', max_iter=-1, probability=True, random_state=None,
      shrinking=True, tol=0.001, verbose=False),
           fit_params=None, iid='warn', n_jobs=None,
           param_grid=[{'C': [1, 10, 100, 1000], 'kernel': ['linear']}, {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}],
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=0)



    time: 30 s
    


```python
y_pred = clf.predict(X_test)
```

    time: 62 ms
    


```python
print(y_pred)
```

    [0 0 1 1 0 2 1 1 0 1 1 0 0 0 2 0 0 0 1]
    time: 0 ns
    


```python
print(clf.score(X_train,Y_train)*100)
```

    100.0
    time: 125 ms
    


```python
from sklearn.metrics import accuracy_score,confusion_matrix
```

    time: 0 ns
    


```python
accuracy_score(y_pred,Y_test)*100
```




    94.73684210526315



    time: 0 ns
    


```python
confusion_matrix(y_pred,Y_test)
```




    array([[9, 0, 1],
           [0, 7, 0],
           [0, 0, 2]], dtype=int64)



    time: 47 ms
    

### Save the model using "PICKLE" library


```python
import pickle
```

    time: 0 ns
    


```python
pickle.dump(clf,open('img_model.p','wb'))
```

    time: 78 ms
    


```python
model = pickle.load(open('img_model.p','rb'))
```

    time: 297 ms
    

# Testing a brand new image urls from google itself


```python
flat_data = []
url = input("Enter the URL")
img = imread(url)
img_resized = resize(img,(150,150,3))
flat_data.append(img_resized.flatten())
flat_data = np.array(flat_data)
print(img.shape)
plt.imshow(img_resized)
y_out = model.predict(flat_data)
y_out = CATEGORIES[y_out[0]]
# formatted string
print(f' Predicted output: {y_out}')
```

    Enter the URLhttps://3.imimg.com/data3/GM/GA/MY-3734274/rugby-ball-250x250.jpg
    

    F:\anaconda\lib\site-packages\skimage\transform\_warps.py:105: UserWarning: The default mode, 'constant', will be changed to 'reflect' in skimage 0.15.
      warn("The default mode, 'constant', will be changed to 'reflect' in "
    F:\anaconda\lib\site-packages\skimage\transform\_warps.py:110: UserWarning: Anti-aliasing will be enabled by default in skimage 0.15 to avoid aliasing artifacts when down-sampling images.
      warn("Anti-aliasing will be enabled by default in skimage 0.15 to "
    

    (250, 250, 3)
     Predicted output: rugby ball leather
    


![png](output_39_3.png)


    time: 4.59 s
    
