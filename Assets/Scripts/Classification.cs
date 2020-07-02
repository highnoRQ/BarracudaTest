using UnityEngine;
using UnityEngine.UI;
using Unity.Barracuda;  // import必須

public class Classification : MonoBehaviour
{
    //  Barracuda 推論用
    public NNModel modelAsset;
    private Model m_RuntimeModel;
    private IWorker m_worker;

    public RenderTexture targetTexture;
    public Text targetText;

    // Start is called before the first frame update
    void Start()
    {
        m_RuntimeModel = ModelLoader.Load(modelAsset);

        var workerType = WorkerFactory.Type.Compute; // GPUで実行する場合はこちらを利用
        // var workerType = WorkerFactory.Type.CSharp;  // CPUで実行する場合はこちらを利用

        m_worker = WorkerFactory.CreateWorker(workerType, m_RuntimeModel);
    }

    private void Update()
    {
        Tensor input = new Tensor(targetTexture);    // RenderTextureから直接Tensorを作成可能
        Inference(input);
        input.Dispose();
    }

    private void Inference(Tensor input)
    {
        m_worker.Execute(input);
        Tensor output = m_worker.PeekOutput();

        var outputArray = output.ToReadOnlyArray();
        int maxIndex = 0;
        float max = 0;
        for (int i = 0; i < outputArray.Length; i++)
        {
            if(max < outputArray[i])
            {
                max = outputArray[i];
                maxIndex = i;
            }
        }

        targetText.text = Library.getImageNetSynset()[maxIndex];

        output.Dispose();   //各ステップごとにTensorは破棄する必要がある(メモリリーク回避のため)
    }

    private void OnDestroy()
    {
        m_worker.Dispose(); //終了時に破棄する
    }

}
