import com.google.gson.GsonBuilder;
import com.google.gson.annotations.SerializedName;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Properties;
import software.amazon.ai.mms.servingsdk.Context;
import software.amazon.ai.mms.servingsdk.ModelServerEndpoint;
import software.amazon.ai.mms.servingsdk.annotations.Endpoint;
import software.amazon.ai.mms.servingsdk.annotations.helpers.EndpointTypes;
import software.amazon.ai.mms.servingsdk.http.Request;
import software.amazon.ai.mms.servingsdk.http.Response;

// The modified endpoint source code for the jar used in this container.
// You can create this endpoint by moving it by cloning the MMS repo:
// > git clone https://github.com/awslabs/mxnet-model-server.git
//
// Copy this file into plugins/endpoints/src/main/java/software/amazon/ai/mms/plugins/endpoints/
// and then from the plugins directory, run:
//
// > ./gradlew fJ
//
// The jar should be available in plugins/build/libs as endpoints-1.0.jar
@Endpoint(
        urlPattern = "execution-parameters",
        endpointType = EndpointTypes.INFERENCE,
        description = "Execution parameters endpoint")
public class ExecutionParameters extends ModelServerEndpoint {

    @Override
    public void doGet(Request req, Response rsp, Context ctx) throws IOException {
        Properties prop = ctx.getConfig();
        SagemakerXgboostResponse r = new SagemakerXgboostResponse();
        r.setMaxConcurrentTransforms(Integer.parseInt(prop.getProperty("NUM_WORKERS", "1")));
        r.setBatchStrategy("MULTI_RECORD");
        r.setMaxPayloadInMB(Integer.parseInt(prop.getProperty("SAGEMAKER_MAX_REQUEST_SIZE", "6")));
        rsp.getOutputStream()
                .write(
                        new GsonBuilder()
                                .setPrettyPrinting()
                                .create()
                                .toJson(r)
                                .getBytes(StandardCharsets.UTF_8));
    }

    /** Response for Model server endpoint */
    public static class SagemakerXgboostResponse {
        @SerializedName("MaxConcurrentTransforms")
        private int maxConcurrentTransforms;

        @SerializedName("BatchStrategy")
        private String batchStrategy;

        @SerializedName("MaxPayloadInMB")
        private int maxPayloadInMB;

        public SagemakerXgboostResponse() {
            maxConcurrentTransforms = 4;
            batchStrategy = "MULTI_RECORD";
            maxPayloadInMB = 6;
        }

        public int getMaxConcurrentTransforms() {
            return maxConcurrentTransforms;
        }

        public String getBatchStrategy() {
            return batchStrategy;
        }

        public int getMaxPayloadInMB() {
            return maxPayloadInMB;
        }

        public void setMaxConcurrentTransforms(int newMaxConcurrentTransforms) {
            maxConcurrentTransforms = newMaxConcurrentTransforms;
        }

        public void setBatchStrategy(String newBatchStrategy) {
            batchStrategy = newBatchStrategy;
        }

        public void setMaxPayloadInMB(int newMaxPayloadInMB) {
            maxPayloadInMB = newMaxPayloadInMB;
        }
    }
}