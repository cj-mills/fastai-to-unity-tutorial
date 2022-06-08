using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using UnityEngine.Rendering;
using System;
using UnityEngine.UI;

public class ImageClassifier : MonoBehaviour
{
    [Header("Scene Objects")]
    [Tooltip("The Screen object for the scene")]
    public Transform screen;

    [Header("Data Processing")]
    [Tooltip("The target minimum model input dimensions")]
    public int targetDim = 216;
    [Tooltip("The compute shader for GPU processing")]
    public ComputeShader processingShader;
    [Tooltip("The material with the fragment shader for GPU processing")]
    public Material processingMaterial;
    
    [Header("Barracuda")]
    [Tooltip("The Barracuda/ONNX asset file")]
    public NNModel modelAsset;
    [Tooltip("The name for the custom softmax output layer")]
    public string softmaxLayer = "softmaxLayer";
    [Tooltip("The name for the custom softmax output layer")]
    public string argmaxLayer = "argmaxLayer";
    [Tooltip("The target output layer index")]
    public int outputLayerIndex = 0;
    [Tooltip("EXPERIMENTAL: Indicate whether to order tensor data channels first")]
    public bool useNCHW = true;
    [Tooltip("The model execution backend")]
    public WorkerFactory.Type workerType = WorkerFactory.Type.Auto;

    [Header("Output Processing")]
    [Tooltip("Asynchronously download model output from the GPU to the CPU.")]
    public bool useAsyncGPUReadback = true;
    [Tooltip("A json file containing the class labels")]
    public TextAsset classLabels;

    [Header("Debugging")]
    [Tooltip("Print debugging messages to the console")]
    public bool printDebugMessages = true;

    [Header("Webcam")]
    [Tooltip("Use a webcam as input")]
    public bool useWebcam = false;
    [Tooltip("The requested webcam dimensions")]
    public Vector2Int webcamDims = new Vector2Int(1280, 720);
    [Tooltip("The requested webcam framerate")]
    [Range(0, 60)]
    public int webcamFPS = 60;

    [Header("GUI")]
    [Tooltip("Display predicted class")]
    public bool displayPredictedClass = true;
    [Tooltip("Display fps")]
    public bool displayFPS = true;
    [Tooltip("The on-screen text color")]
    public Color textColor = Color.red;
    [Tooltip("The scale value for the on-screen font size")]
    [Range(0, 99)]
    public int fontScale = 50;
    [Tooltip("The number of seconds to wait between refreshing the fps value")]
    [Range(0.01f, 1.0f)]
    public float fpsRefreshRate = 0.1f;
    [Tooltip("The toggle for using a webcam as the input source")]
    public Toggle useWebcamToggle;
    [Tooltip("The dropdown menu that lists available webcam devices")]
    public Dropdown webcamDropdown;

    // List of available webcam devices
    private WebCamDevice[] webcamDevices;
    // Live video input from a webcam
    private WebCamTexture webcamTexture;
    // The name of the current webcam  device
    private string currentWebcam;

    // The test image dimensions
    private Vector2Int imageDims;
    // The test image texture
    private Texture imageTexture;
    // The current screen object dimensions
    private Vector2Int screenDims;
    // The model input texture
    private RenderTexture inputTexture;

    // The main interface to execute models
    private IWorker engine;
    // Stores the input data for the model
    private Tensor input;
    
    // Stores the raw model output on the GPU when using useAsyncGPUReadback
    private RenderTexture outputTextureGPU;
    // Stores the raw model output on the CPU when using useAsyncGPUReadback
    private Texture2D outputTextureCPU;

    // A class for reading in class labels from a JSON file
    class ClassLabels { public string[] classes; }
    // The ordered list of class names
    private string[] classes;
    // Stores the predicted class index
    private int classIndex;

    // The current frame rate value
    private int fps = 0;
    // Controls when the frame rate value updates
    private float fpsTimer = 0f;
    

    /// <summary>
    /// Initialize the selected webcam device
    /// </summary>
    /// <param name="deviceName">The name of the selected webcam device</param>
    private void InitializeWebcam(string deviceName)
    {
        // Stop any webcams already playing
        if (webcamTexture && webcamTexture.isPlaying) webcamTexture.Stop();

        // Create a new WebCamTexture
        webcamTexture = new WebCamTexture(deviceName, webcamDims.x, webcamDims.y, webcamFPS);

        // Start the webcam
        webcamTexture.Play();
        // Check if webcam is playing
        useWebcam = webcamTexture.isPlaying;
        // Update toggle value
        useWebcamToggle.SetIsOnWithoutNotify(useWebcam);

        Debug.Log(useWebcam ? "Webcam is playing" : "Webcam not playing, option disabled");
    }


    /// <summary>
    /// Resize and position an in-scene screen object
    /// </summary>
    private void InitializeScreen()
    {
        // Set the texture for the screen object
        screen.gameObject.GetComponent<MeshRenderer>().material.mainTexture = useWebcam ? webcamTexture : imageTexture;
        // Set the screen dimensions
        screenDims = useWebcam ? new Vector2Int(webcamTexture.width, webcamTexture.height) : imageDims;

        // Flip the screen around the Y-Axis when using webcam
        float yRotation = useWebcam ? 180f : 0f;
        // Invert the scale value for the Z-Axis when using webcam
        float zScale = useWebcam ? -1f : 1f;

        // Set screen rotation
        screen.rotation = Quaternion.Euler(0, yRotation, 0);
        // Adjust the screen dimensions
        screen.localScale = new Vector3(screenDims.x, screenDims.y, zScale);

        // Adjust the screen position
        screen.position = new Vector3(screenDims.x / 2, screenDims.y / 2, 1);
    }


    /// <summary>
    /// Initialize the GUI dropdown list
    /// </summary>
    private void InitializeDropdown()
    {
        // Create list of webcam device names
        List<string> webcamNames = new List<string>();
        foreach(WebCamDevice device in webcamDevices) webcamNames.Add(device.name);

        // Remove default dropdown options
        webcamDropdown.ClearOptions();
        // Add webcam device names to dropdown menu
        webcamDropdown.AddOptions(webcamNames);
        // Set the value for the dropdown to the current webcam device
        webcamDropdown.SetValueWithoutNotify(webcamNames.IndexOf(currentWebcam));
    }


    /// <summary>
    /// Resize and position the main camera based on an in-scene screen object
    /// </summary>
    /// <param name="screenDims">The dimensions of an in-scene screen object</param>
    private void InitializeCamera(Vector2Int screenDims, string cameraName = "Main Camera")
    {
        // Get a reference to the Main Camera GameObject
        GameObject camera = GameObject.Find(cameraName);
        // Adjust the camera position to account for updates to the screenDims
        camera.transform.position = new Vector3(screenDims.x / 2, screenDims.y / 2, -10f);
        // Render objects with no perspective (i.e. 2D)
        camera.GetComponent<Camera>().orthographic = true;
        // Adjust the camera size to account for updates to the screenDims
        camera.GetComponent<Camera>().orthographicSize = screenDims.y / 2;
    }


    /// <summary>
    /// Initialize an interface to execute the specified model using the specified backend
    /// </summary>
    /// <param name="model">The target model representation</param>
    /// <param name="workerType">The target compute backend</param>
    /// <param name="useNCHW">EXPERIMENTAL: The channel order for the compute backend</param>
    /// <returns></returns>
    private IWorker InitializeWorker(Model model, WorkerFactory.Type workerType, bool useNCHW = true)
    {
        // Validate the selected worker type
        workerType = WorkerFactory.ValidateType(workerType);

        // Set the channel order of the compute backend to channel-first
        if (useNCHW) ComputeInfo.channelsOrder = ComputeInfo.ChannelsOrder.NCHW;

        // Create a worker to execute the model using the selected backend
        return WorkerFactory.CreateWorker(workerType, model);
    }


    // Start is called before the first frame update
    void Start()
    {
        // Get the source image texture
        imageTexture = screen.gameObject.GetComponent<MeshRenderer>().material.mainTexture;
        // Get the source image dimensions as a Vector2Int
        imageDims = new Vector2Int(imageTexture.width, imageTexture.height);

        // Initialize list of available webcam devices
        webcamDevices = WebCamTexture.devices;
        foreach (WebCamDevice device in webcamDevices) Debug.Log(device.name);
        currentWebcam = webcamDevices[0].name;
        useWebcam = webcamDevices.Length > 0 ? useWebcam : false;
        // Initialize webcam
        if (useWebcam) InitializeWebcam(currentWebcam);

        // Resize and position the screen object using the source image dimensions
        InitializeScreen();
        // Resize and position the main camera using the source image dimensions
        InitializeCamera(screenDims);

        // Get an object oriented representation of the model
        Model m_RunTimeModel = ModelLoader.Load(modelAsset);
        // Get the name of the target output layer
        string outputLayer = m_RunTimeModel.outputs[outputLayerIndex];

        // Create a model builder to modify the m_RunTimeModel
        ModelBuilder modelBuilder = new ModelBuilder(m_RunTimeModel);

        // Add a new Softmax layer
        modelBuilder.Softmax(softmaxLayer, outputLayer);
        // Add a new Argmax layer
        modelBuilder.Reduce(Layer.Type.ArgMax, argmaxLayer, softmaxLayer);
        // Initialize the interface for executing the model
        engine = InitializeWorker(modelBuilder.model, workerType, useNCHW);

        // Initialize the GPU output texture
        outputTextureGPU = RenderTexture.GetTemporary(1, 1, 24, RenderTextureFormat.ARGBHalf);
        // Initialize the CPU output texture
        outputTextureCPU = new Texture2D(1, 1, TextureFormat.RGBAHalf, false);
                
        // Initialize list of class labels from JSON file
        classes = JsonUtility.FromJson<ClassLabels>(classLabels.text).classes;

        // Initialize the webcam dropdown list
        InitializeDropdown();
    }


    /// <summary>
    /// Process the provided image using the specified function on the GPU
    /// </summary>
    /// <param name="image">The target image RenderTexture</param>
    /// <param name="computeShader">The target ComputerShader</param>
    /// <param name="functionName">The target ComputeShader function</param>
    /// <returns></returns>
    private void ProcessImageGPU(RenderTexture image, ComputeShader computeShader, string functionName)
    {
        // Specify the number of threads on the GPU
        int numthreads = 8;
        // Get the index for the specified function in the ComputeShader
        int kernelHandle = computeShader.FindKernel(functionName);
        // Define a temporary HDR RenderTexture
        RenderTexture result = RenderTexture.GetTemporary(image.width, image.height, 24, RenderTextureFormat.ARGBHalf);
        // Enable random write access
        result.enableRandomWrite = true;
        // Create the HDR RenderTexture
        result.Create();

        // Set the value for the Result variable in the ComputeShader
        computeShader.SetTexture(kernelHandle, "Result", result);
        // Set the value for the InputImage variable in the ComputeShader
        computeShader.SetTexture(kernelHandle, "InputImage", image);

        // Execute the ComputeShader
        computeShader.Dispatch(kernelHandle, result.width / numthreads, result.height / numthreads, 1);

        // Copy the result into the source RenderTexture
        Graphics.Blit(result, image);

        // Release the temporary RenderTexture
        RenderTexture.ReleaseTemporary(result);
    }


    /// <summary>
    /// Scale the source image resolution to the target input dimensions
    /// while maintaing the source aspect ratio.
    /// </summary>
    /// <param name="imageDims"></param>
    /// <param name="targetDims"></param>
    /// <returns></returns>
    private Vector2Int CalculateInputDims(Vector2Int imageDims, int targetDim)
    {
        // Clamp the minimum dimension value to 64px
        targetDim = Mathf.Max(targetDim, 64);

        Vector2Int inputDims = new Vector2Int();

        // Calculate the input dimensions using the target minimum dimension
        if (imageDims.x >= imageDims.y)
        {
            inputDims[0] = (int)(imageDims.x / ((float)imageDims.y / (float)targetDim));
            inputDims[1] = targetDim;
        }
        else
        {
            inputDims[0] = targetDim;
            inputDims[1] = (int)(imageDims.y / ((float)imageDims.x / (float)targetDim));
        }

        return inputDims;
    }


    /// <summary>
    /// Called once AsyncGPUReadback has been completed
    /// </summary>
    /// <param name="request"></param>
    private void OnCompleteReadback(AsyncGPUReadbackRequest request)
    {
        if (request.hasError)
        {
            Debug.Log("GPU readback error detected.");
            return;
        }

        // Make sure the Texture2D is not null
        if (outputTextureCPU)
        {
            // Fill Texture2D with raw data from the AsyncGPUReadbackRequest
            outputTextureCPU.LoadRawTextureData(request.GetData<uint>());
            // Apply changes to Textur2D
            outputTextureCPU.Apply();
        }
    }


    /// <summary>
    /// Process the raw model output to get the predicted class index
    /// </summary>
    /// <param name="engine">The interface for executing the model</param>
    /// <returns></returns>
    private int ProcessOutput(IWorker engine)
    {
        int classIndex = -1;

        // Get raw model output
        Tensor output = engine.PeekOutput(argmaxLayer);

        if (useAsyncGPUReadback)
        {
            // Copy model output to a RenderTexture
            output.ToRenderTexture(outputTextureGPU);
            // Asynchronously download model output from the GPU to the CPU
            AsyncGPUReadback.Request(outputTextureGPU, 0, TextureFormat.RGBAHalf, OnCompleteReadback);
            // Get the predicted class index
            classIndex = (int)outputTextureCPU.GetPixel(0, 0).r;

            // Check if index is valid
            if (classIndex < 0 || classIndex >= classes.Length) Debug.Log("Output texture not initialized");
        }
        else
        {
            // Get the predicted class index
            classIndex = (int)output[0];
        }

        if (printDebugMessages) Debug.Log($"Class Index: {classIndex}");

        // Dispose Tensor and associated memories.
        output.Dispose();

        return classIndex;
    }
    

    // Update is called once per frame
    void Update()
    {
        useWebcam = webcamDevices.Length > 0 ? useWebcam : false;
        if (useWebcam)
        {
            // Initialize webcam if it is not already playing
            if (!webcamTexture || !webcamTexture.isPlaying) InitializeWebcam(currentWebcam);

            // Skip the rest of the method if the webcam is not initialized
            if (webcamTexture.width <= 16) return;

            // Make sure screen dimensions match webcam resolution when using webcam
            if (screenDims.x != webcamTexture.width)
            {
                // Resize and position the screen object using the source image dimensions
                InitializeScreen();
                // Resize and position the main camera using the source image dimensions
                InitializeCamera(screenDims);
            }
        }
        else if (webcamTexture && webcamTexture.isPlaying)
        {
            // Stop the current webcam
            webcamTexture.Stop();

            // Resize and position the screen object using the source image dimensions
            InitializeScreen();
            // Resize and position the main camera using the source image dimensions
            InitializeCamera(screenDims);
        }

        // Scale the source image resolution
        Vector2Int inputDims = CalculateInputDims(screenDims, targetDim);
        if (printDebugMessages) Debug.Log($"Input Dims: {inputDims.x} x {inputDims.y}");

        // Initialize the input texture with the calculated input dimensions
        inputTexture = RenderTexture.GetTemporary(inputDims.x, inputDims.y, 24, RenderTextureFormat.ARGBHalf);
        if (printDebugMessages) Debug.Log($"Input Dims: {inputTexture.width}x{inputTexture.height}");

        // Copy the source texture into model input texture
        Graphics.Blit((useWebcam ? webcamTexture : imageTexture), inputTexture);

        // Disable asynchronous GPU readback when not using a Compute Shader backend
        useAsyncGPUReadback = engine.Summary().Contains("Unity.Barracuda.ComputeVarsWithSharedModel") ? useAsyncGPUReadback : false;

        if (SystemInfo.supportsComputeShaders)
        {
            // Normalize the input pixel data
            ProcessImageGPU(inputTexture, processingShader, "NormalizeImageNet");

            // Initialize a Tensor using the inputTexture
            input = new Tensor(inputTexture, channels: 3);
        }
        else
        {
            // Define a temporary HDR RenderTexture
            RenderTexture result = RenderTexture.GetTemporary(inputTexture.width,
                inputTexture.height, 24, RenderTextureFormat.ARGBHalf);
            RenderTexture.active = result;

            // Apply preprocessing steps
            Graphics.Blit(inputTexture, result, processingMaterial);

            // Initialize a Tensor using the inputTexture
            input = new Tensor(result, channels: 3);
            RenderTexture.ReleaseTemporary(result);
        }

        // Execute the model with the input Tensor
        engine.Execute(input);
        // Dispose Tensor and associated memories.
        input.Dispose();

        // Release the input texture
        RenderTexture.ReleaseTemporary(inputTexture);
        // Get the predicted class index
        classIndex = ProcessOutput(engine);
        // Check if index is valid
        bool validIndex = classIndex >= 0 && classIndex < classes.Length;
        if (printDebugMessages) Debug.Log(validIndex ? $"Predicted Class: {classes[classIndex]}" : "Invalid index");

        // Unload assets when running in a web browser
        if (Application.platform == RuntimePlatform.WebGLPlayer) Resources.UnloadUnusedAssets();
    }


    /// <summary>
    /// This method is called when the value for the webcam toggle changes
    /// </summary>
    /// <param name="useWebcam"></param>
    public void UpdateWebcamToggle(bool useWebcam)
    {
        this.useWebcam = useWebcam;
    }


    /// <summary>
    /// The method is called when the selected value for the webcam dropdown changes
    /// </summary>
    public void UpdateWebcamDevice()
    {
        currentWebcam = webcamDevices[webcamDropdown.value].name;
        Debug.Log($"Selected Webcam: {currentWebcam}");
        // Initialize webcam if it is not already playing
        if (useWebcam) InitializeWebcam(currentWebcam);

        // Resize and position the screen object using the source image dimensions
        InitializeScreen();
        // Resize and position the main camera using the source image dimensions
        InitializeCamera(screenDims);
    }


    // OnGUI is called for rendering and handling GUI events.
    public void OnGUI()
    {
        // Define styling information for GUI elements
        GUIStyle style = new GUIStyle
        {
            fontSize = (int)(Screen.width * (1f / (100f - fontScale)))
        };
        style.normal.textColor = textColor;

        // Define screen spaces for GUI elements
        Rect slot1 = new Rect(10, 10, 500, 500);
        Rect slot2 = new Rect(10, style.fontSize * 1.5f, 500, 500);

        // Verify predicted class index is valid
        bool validIndex = classIndex >= 0 && classIndex < classes.Length;
        string content = $"Predicted Class: {(validIndex ? classes[classIndex] : "Invalid index")}";
        if (displayPredictedClass) GUI.Label(slot1, new GUIContent(content), style);

        // Update framerate value
        if (Time.unscaledTime > fpsTimer)
        {
            fps = (int)(1f / Time.unscaledDeltaTime);
            fpsTimer = Time.unscaledTime + fpsRefreshRate;
        }

        // Adjust screen position when not showing predicted class
        Rect fpsRect = displayPredictedClass ? slot2 : slot1;
        if (displayFPS) GUI.Label(fpsRect, new GUIContent($"FPS: {fps}"), style);
    }


    // OnDisable is called when the MonoBehavior becomes disabled
    private void OnDisable()
    {
        // Release the resources allocated for the outputTextureGPU
        RenderTexture.ReleaseTemporary(outputTextureGPU);

        // Release the resources allocated for the inference engine
        engine.Dispose();
    }
}
