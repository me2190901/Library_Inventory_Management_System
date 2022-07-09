<h2>Objectives</h2>
<ol>
    <li>Building a scene text reading system designed explicitly for book spine reading and library
        inventory management.</li>
    <li>Developing a deep sequential labeling model based on convolutional neural nets (CNN) and
        recurrent neural nets (RNN) for text recognition to automatically localize, recognize and
        index the text on bookshelves images.</li>
    <li>Building a digital book inventory for processing of bookshelves images to localize and
        recognize book spine text
    <li>Employing a per-timestep classification loss associated with a weighted Connectionist
        Temporal Classification (CTC) loss function to accelerate training and improve
        performance.
    <li>Deployment of the system through a website is easy to use and easily scalable to employ
        directly in the central library.
</ol>
<h2> Approach</h2>
<ol>
    <li>
        <h3>Text Localization</h3>
    </li>
    Segmentation of the book spine is a critical component of our system since every book is
    expected to be indexed and queried independently. So, we first segment each book spine image
    and then localize text on these images. The path for attaining text localization is as follow:
    <ol>
        <li>Pre-processing the image by resizing it, applying Gaussian blur, resizing,
            BGR2GRAY, etc., and finding the dominant direction of book spines by Hough
            Transformation.
        <li>Rotation of image based on dominant direction obtained from pre-processing.
        <li>Detecting edges of the image with the help of Canny Edge Detection.
        <li>Remove Short Clusters and Connected Components in Edge Detected Image.
        <li>Separated Spines by detecting dominant directions and applying Hough.
        <li>Separated Spines by detecting dominant directions and applying Hough
            Transformation.
            Transformation.
    </ol>
    <li>
        <h3>Text Recognition</h3>
    </li>
    In our system, book spine images are identified based on the recognized text, then used for
    indexing and searching from a book database.
    <ol>
        <li><b>Text Recognition via Sequence Labelling.</b></li>
        We cast a sequential labeling task for text recognition to recognize a sequence of
        characters simultaneously rather than deploying a conventional approach of first segmenting
        and identifying each character, then predicting a word based on a language model.
        Traditional methods are susceptible to various distortions in images causing character-level
        segmentation imperfections. The process to deploying this is as follows:
        <ol>
            <li>First of all, extracting features from the image [Uploading SURA_FINAL_2.pdf…]()
                using a sequence of deep CNN say F =
                {f1, f2, · · ·, fT}.
            <li>Employing a 'Bidirectional Long Short-Term Memory (B-LSTM)' over an extracted
                CNN features to exploit further the interdependence among features yielding another
                sequence X = {x1, x2, · · ·, xT} as final outputs.
            <li>Normalize exploited features and then interpret them as the emission of a character or a
                blank label at a specific timestep.
            <li>The target word can also be viewed as a sequence of characters: Y = {y1, y2, · · ·, yL}.
                Since sequences 'X' and 'Y' have different lengths, we adopt CTC loss to train an RNN.
            <li>Forward-backward dynamic programming methods can be used to compute the
                gradient of the CTC loss efficiently.
            <li>Beam searching can be applied to find the most likely Y from the output sequence X.
        </ol>
        <li><b> CTC Training with Per-Timestep Supervision.</b></li>
        <p>The Black labels will typically dominate the output sequence during the CTC training, and
            non-blank labels only appear as isolated peaks.</p>
        <p>All paths have similar probabilities at the early stage of CTC training, where model weights
            are randomly initialized. As we add a blank label between each character, there are more
            possible paths going through a blank label at a given timestep in the CTC forward-backward
            graph. As a result, the probability of a given timestep being an empty label is much higher
            than any other label when summing up all valid paths in the CTC graph. So, we have to
            accelerate our training as it takes many iterations for non-black labels to appear in the output
            sequence.</p>
        <p>To accelerate our training, we introduced per-timestep supervision in our project, i.e., If
            character-level bounding boxes are available, we can decide the label of xi at each time step
            i, based on its receptive field.</p>
    </ol>
    <li>
        <h3>Dataset Search and Checklist.<h3>
    </li>
    After the successful localization and recognition of the words from the above-proposed pipeline, the
    following algorithm was used to implement the checklist feature:
    <ol>
        <li>The collected words were first queried in a spell checker to get a sense of the results
            obtained.
        </li>
        <li>Weighted probabilities were used to find the most likely title while searching each word in
            the database. Emphasis was given on the consolidated length of the word as longer words
            will indicate more chances of producing nearby results.
        </li>
        <li>The most probable title is searched thoroughly in the database based on the weighted
            probability.</li>
        <li>Finally, the most probable title index is searched, and many books are appended to the
            checklist.
        </li>
    </ol>
    <li>
        <h3>Model Deployment</h3>
    </li>
    The above infrastructure is deployed through a website that integrates a python deep learning model
    with HTML webpages using the Flask web framework. Bootstrap libraries were used in the website
    to enhance the user experience. Bootstrap also helped in improving the user interface to make the
    website handy to use. The website has a landing page that takes a single image as input to run the
    model. The navigation bar at the top links to the home page of IITD, the model deployment landing
    page, and the project page. The input image can be browsed from the local system and will be used
    for initiating the model.
</ol>
<h2>Results</h2>
<li><b>Text Localization:</b> We have used OpenCV to perform Text Localization, giving us up to
    <b>97.2%</b> accuracy of segmenting spines and localizing text on these spines. Results are shown
    in the figure below.
</li>
<li><b>Text Recognition:</b> For text recognition, as we know that training Deep Learning
    CRNN models require high GPU performance and very high knowledge of mathematics, we
    used a pre-trained model based on [2] as it gives us the highest accuracy when we run the
    model on several benchmark datasets like IC03, SVT, and III5K and compared with several
    other models using a standard evaluation protocol [3]. Results are shown in the table below.
</li>
<li><b>Dataset search and checklist:</b> We are running queries using pandas to search each
    word in the dataset (.csv file) and assigning probabilities to each title. After finding the title,
    we append many books retrieved in the dataset. Even if we searched the title wrong, we
    could use human intervention in API to correct them. Making our model more accurate.</li>
<br>
<b>The scope of improvement we have recognized in this model is:</b>
<li>We have to crop our image to remove the extra part that does not contain the spines
    boundary. Auto-cropping algorithms were experimented with but resulted in subpar
    results. Separate infrastructure needs to be developed to overcome the issue of pre-cropped images.</li>
<li>Further robust searching algorithms can be employed using frameworks like Apache
    Solr[10] to make a powerful search engine for title search in the dataset.
    <h2>
        Discussion
    </h2>
    <p>
        Previous work on book inventory management has typically focused on book spine detection and
        retrievals, such as the framework for high-frequency filtering and thresholding. The performance of
        most of the existing approaches is limited by book spine segmentation and off-the-shelf OCR
        systems. In most of these approaches the handcrafted features-based book spine segmentation
        suffers from image distortion and low contrast between books. Recently, scene text reading has
        become popular in computer vision. Here we presented a deep neural network-based system that
        reads scene text and shows that scene text reading can be effectively utilized for book inventories
        management and book retrieval. Our system achieved robust performance on book retrieval tasks
        combined with other image processing techniques such as Hough Transform
    </p>
    <p>
        The deployment of the model through the website makes the work presented here easily deployable
        by scaling as per the community's needs. The websites contain an integrated checklist that gives
        real-time results upon uploading the image in a table that gives the number of books detected in the
        image, hence solving the purpose of inventory management and audit.
    </p>
    <p>
        The website comes with a feature of human intervention that has not been used yet in any research
        work. In the lower half of the web page containing the tabled checklist, text boxes can input the
        books that are not detected from the image. We can also input the number of books along with the
        name. This feature allows the addition of a book if it is not detected through the model. We can also
        delete a book from the checklist. We can add up to 20 book titles and their number at a time, and
        this input through the website can be directly updated in the base containing the checklist.
    </p>
    <p>
        Although this model can be improved by deploying auto-cropping tool and strong search engine,
        this model is giving sufficient results to be deployed in large citadels like Central Library, IIT
        Delhi.
    </p>
