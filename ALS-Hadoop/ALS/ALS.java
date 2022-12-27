package ALS;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.Iterator;
import java.util.Random;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.filecache.DistributedCache;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.map.MultithreadedMapper;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirIterable;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirValueIterator;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.similarity.cooccurrence.Vectors;
import org.apache.mahout.math.map.OpenIntObjectHashMap;
import org.mortbay.log.Log;

import com.google.common.base.Charsets;
import com.google.common.io.Closeables;


public class ALS {
	private static int numFeatures = 20;
	private static int numIterations = 10;
	private static int numThreadsPerSolver = 2;
	private static double lambda = 0.065;
	private static String dirName = "ml-100k";
	private static String root = "hdfs://localhost:9000/als/" + dirName;
	
	
	private static long runJob(Job job) throws Exception {
		long start_time = System.currentTimeMillis();
		boolean flag = job.waitForCompletion(true);
		long end_time = System.currentTimeMillis();
		long ret = flag ? end_time - start_time : -1 ;
		return ret;
	}
	
	private static Job createJob(Configuration conf, FileSystem fs, Path inputPath, Path outputPath) throws Exception {
		Job job = Job.getInstance(conf, "ALS");
		job.setJarByClass(ALS.class);
		
		FileInputFormat.setInputPaths(job, inputPath);
		FileOutputFormat.setOutputPath(job, outputPath);
		
		if (fs.exists(outputPath)) {
			fs.delete(outputPath, true);
		}
		
		return job;
	}
	
	static class ItemRatingVectorsMapper extends Mapper<LongWritable, Text, IntWritable, VectorWritable> {
		private final IntWritable itemIDWritable = new IntWritable();
		private final VectorWritable ratingsWritable = new VectorWritable(true);
		private final Vector ratings = new RandomAccessSparseVector(Integer.MAX_VALUE, 1);

		@Override
		protected void map(LongWritable offset, Text line, Context ctx) throws IOException, InterruptedException {
//			String[] tokens = line.toString().split("\\s+");
			String[] tokens = line.toString().split(",");
			int userID = Integer.parseInt(tokens[0]);
			int itemID = Integer.parseInt(tokens[1]);
			float rating = Float.parseFloat(tokens[2]);

			ratings.setQuick(userID, rating);

			itemIDWritable.set(itemID);
			ratingsWritable.set(ratings);

			ctx.write(itemIDWritable, ratingsWritable);

			// prepare instance for reuse
			ratings.setQuick(userID, 0.0d);
		}
	}
	
	static class UserRatingVectorsMapper extends Mapper<LongWritable, Text, IntWritable, VectorWritable> {
		private final IntWritable userIDWritable = new IntWritable();
		private final VectorWritable ratingsWritable = new VectorWritable(true);
		private final Vector ratings = new RandomAccessSparseVector(Integer.MAX_VALUE, 1);

		@Override
		protected void map(LongWritable offset, Text line, Context ctx) throws IOException, InterruptedException {
//			String[] tokens = line.toString().split("\\s+");
			String[] tokens = line.toString().split(",");
			int userID = Integer.parseInt(tokens[0]);
			int itemID = Integer.parseInt(tokens[1]);
			float rating = Float.parseFloat(tokens[2]);

			ratings.setQuick(itemID, rating);

			userIDWritable.set(itemID);
			ratingsWritable.set(ratings);

			ctx.write(userIDWritable, ratingsWritable);

			// prepare instance for reuse
			ratings.setQuick(userID, 0.0d);
		}
	}
	
	static class VectorSumCombiner extends Reducer<WritableComparable<?>, VectorWritable, WritableComparable<?>, VectorWritable> {
		private final VectorWritable result = new VectorWritable();
		
		@Override
		protected void reduce(WritableComparable<?> key, Iterable<VectorWritable> values, Context ctx) throws IOException, InterruptedException {
			result.set(Vectors.sum(values.iterator()));
			ctx.write(key, result);
		}
	}
	
	static class VectorSumReducer extends Reducer<WritableComparable<?>, VectorWritable, WritableComparable<?>, VectorWritable> {
		private final VectorWritable result = new VectorWritable();
		@Override
		protected void reduce(WritableComparable<?> key, Iterable<VectorWritable> values, Context ctx) throws IOException, InterruptedException {
			Vector sum = Vectors.sum(values.iterator());
			result.set(new SequentialAccessSparseVector(sum));
			ctx.write(key, result);
		}
	}
	
	static class AverageRatingMapper extends Mapper<IntWritable,VectorWritable,IntWritable,VectorWritable> {
		private final IntWritable firstIndex = new IntWritable(0);
		private final Vector featureVector = new RandomAccessSparseVector(Integer.MAX_VALUE, 1);
		private final VectorWritable featureVectorWritable = new VectorWritable();

		@Override
		protected void map(IntWritable r, VectorWritable v, Context ctx) throws IOException, InterruptedException {
			RunningAverage avg = new FullRunningAverage();
			for (Vector.Element e : v.get().nonZeroes()) {
				avg.addDatum(e.get());
			}

			featureVector.setQuick(r.get(), avg.getAverage());
			featureVectorWritable.set(featureVector);
			ctx.write(firstIndex, featureVectorWritable);

			// prepare instance for reuse
			featureVector.setQuick(r.get(), 0.0d);
		}
	}
	
	static class TransposeMapper extends Mapper<IntWritable,VectorWritable,IntWritable,VectorWritable> {
		@Override
		protected void map(IntWritable r, VectorWritable v, Context ctx) throws IOException, InterruptedException {
			int row = r.get();
			for (Vector.Element e : v.get().nonZeroes()) {
				RandomAccessSparseVector tmp = new RandomAccessSparseVector(Integer.MAX_VALUE, 1);
				tmp.setQuick(row, e.get());
				r.set(e.index());
				ctx.write(r, new VectorWritable(tmp));
			}
		}
	}
	
	static class MergeUserVectorsReducer extends Reducer<WritableComparable<?>,VectorWritable,WritableComparable<?>,VectorWritable> {
		private final VectorWritable result = new VectorWritable();

		@Override
		public void reduce(WritableComparable<?> key, Iterable<VectorWritable> vectors, Context ctx) throws IOException, InterruptedException {
			Vector merged = VectorWritable.merge(vectors.iterator()).get();
			result.set(new SequentialAccessSparseVector(merged));
			ctx.write(key, result);
//			ctx.getCounter(Stats.NUM_USERS).increment(1);
		}
	}

	static class MergeVectorsCombiner extends Reducer<WritableComparable<?>,VectorWritable,WritableComparable<?>,VectorWritable> {
		@Override
		public void reduce(WritableComparable<?> key, Iterable<VectorWritable> vectors, Context ctx) throws IOException, InterruptedException {
			ctx.write(key, VectorWritable.merge(vectors.iterator()));
		}
	}
	
	static class MergeVectorsReducer extends Reducer<WritableComparable<?>,VectorWritable,WritableComparable<?>,VectorWritable> {
		private final VectorWritable result = new VectorWritable();

		@Override
		public void reduce(WritableComparable<?> key, Iterable<VectorWritable> vectors, Context ctx) throws IOException, InterruptedException {
			Vector merged = VectorWritable.merge(vectors.iterator()).get();
			result.set(new SequentialAccessSparseVector(merged));
			ctx.write(key, result);
		}
	}
	
	@SuppressWarnings("resource")
	public static void main(String[] args) throws Exception {
		Configuration conf = new Configuration();
		
		String input = root + "/ratings.train";
		String output = root + "/AT";
		Path inputPath = new Path(input);
		Path outputPath = new Path(output);
		FileSystem fs = inputPath.getFileSystem(conf);
		Job job;
		long res;
		
		job = createJob(conf, fs, inputPath, outputPath);
		job.setMapperClass(ItemRatingVectorsMapper.class);
		job.setMapOutputKeyClass(IntWritable.class);
		job.setMapOutputValueClass(VectorWritable.class);
		job.setCombinerClass(VectorSumCombiner.class);
		job.setReducerClass(VectorSumReducer.class);
		job.setOutputKeyClass(IntWritable.class);
		job.setOutputValueClass(VectorWritable.class);
		job.setOutputFormatClass(SequenceFileOutputFormat.class);
		res = runJob(job);
		if (res < 0)
			return ;
		int numItems = 0;
		Iterator<VectorWritable> iterator = new SequenceFileDirValueIterator<VectorWritable>(outputPath, PathType.LIST, PathFilters.partFilter(), null, true, conf);
		while (iterator.hasNext()) {
			numItems++;
			iterator.next();
		}
		
		input = root + "/AT";
		inputPath = new Path(input);
		output = root + "/A";
		outputPath = new Path(output);
//		job = createJob(conf, fs, inputPath, outputPath);
//		job.setMapperClass(UserRatingVectorsMapper.class);
//		job.setMapOutputKeyClass(IntWritable.class);
//		job.setMapOutputValueClass(VectorWritable.class);
//		job.setCombinerClass(VectorSumCombiner.class);
//		job.setReducerClass(VectorSumReducer.class);
//		job.setOutputKeyClass(IntWritable.class);
//		job.setOutputValueClass(VectorWritable.class);
//		job.setOutputFormatClass(SequenceFileOutputFormat.class);
		job = createJob(conf, fs, inputPath, outputPath);
		job.setInputFormatClass(SequenceFileInputFormat.class);
		job.setMapperClass(TransposeMapper.class);
		job.setMapOutputKeyClass(IntWritable.class);
		job.setMapOutputValueClass(VectorWritable.class);
		job.setCombinerClass(VectorSumCombiner.class);
		job.setReducerClass(MergeUserVectorsReducer.class);
		job.setOutputKeyClass(IntWritable.class);
		job.setOutputValueClass(VectorWritable.class);
		job.setOutputFormatClass(SequenceFileOutputFormat.class);
		res = runJob(job);
		if (res < 0)
			return ;
		
		int numUsers = 0;
		iterator = new SequenceFileDirValueIterator<VectorWritable>(outputPath, PathType.LIST, PathFilters.partFilter(), null, true, conf);
		while (iterator.hasNext()) {
			numUsers++;
			iterator.next();
		}
	    
//		input = root + "/AT/part-r-00000";
//		inputPath = new Path(input);
		output = root + "/avg";
		outputPath = new Path(output);
		job = createJob(conf, fs, inputPath, outputPath);
		job.setInputFormatClass(SequenceFileInputFormat.class);
		job.setMapperClass(AverageRatingMapper.class);
		job.setMapOutputKeyClass(IntWritable.class);
		job.setMapOutputValueClass(VectorWritable.class);
		job.setCombinerClass(MergeVectorsCombiner.class);
		job.setReducerClass(MergeVectorsReducer.class);
		job.setOutputKeyClass(IntWritable.class);
		job.setOutputValueClass(VectorWritable.class);
		job.setOutputFormatClass(SequenceFileOutputFormat.class);
		res = runJob(job);
		if (res < 0)
			return ;
		
		iterator = new SequenceFileDirValueIterator<VectorWritable>(outputPath, PathType.LIST, PathFilters.partFilter(), null, true, conf);
		Vector averageRatings = iterator.hasNext() ? iterator.next().get() : null;
		int numItems1 = averageRatings.getNumNondefaultElements();

		Log.info("Found {} users and {} items", numUsers, numItems);
		Log.info("Found {} items", numItems1);
		
		initializeM(averageRatings, conf, fs);
		
		Path pathToUserRatings = new Path(root+"/A");
		Path pathToItemRatings = new Path(root+"/AT");
		Path pathToU, pathToM;
		
		for (int currentIteration = 0; currentIteration < numIterations; currentIteration++) {
			/* broadcast M, read A row-wise, recompute U row-wise */
			Log.info("Recomputing U (iteration {}/{})", currentIteration, numIterations);
			pathToU = currentIteration == numIterations - 1 ? new Path(root+"/U") : new Path(root+"/U-"+(currentIteration));
			pathToM = new Path(root+"/M-"+(currentIteration-1));
			res = runSolver(pathToUserRatings, pathToU, pathToM, currentIteration, "U", numItems, conf, fs);
			if (res < 0)
				return ;
			
			/* broadcast U, read A' row-wise, recompute M row-wise */
			Log.info("Recomputing M (iteration {}/{})", currentIteration, numIterations);
			pathToM = currentIteration == numIterations - 1 ? new Path(root+"/M") : new Path(root+"/M-"+(currentIteration));
			res = runSolver(pathToItemRatings, pathToM, pathToU, currentIteration, "M", numUsers, conf, fs);
			if (res < 0)
				return ;
		}
		
		Log.info("Evaluating");
		res = runEvaluator(conf, fs);
	}
	
	@SuppressWarnings("deprecation")
	private static void initializeM(Vector averageRatings, Configuration conf, FileSystem fs) throws IOException {
		Random random = RandomUtils.getRandom();

		SequenceFile.Writer writer = null;
		try {
			writer = new SequenceFile.Writer(fs, conf, new Path(root+"/M--1", "part-m-00000"), IntWritable.class, VectorWritable.class);
			IntWritable index = new IntWritable();
			VectorWritable featureVector = new VectorWritable();
			
			for (Vector.Element e : averageRatings.nonZeroes()) {
				Vector row = new DenseVector(numFeatures);
				row.setQuick(0, e.get());
				for (int m = 1; m < numFeatures; m++) {
					row.setQuick(m, random.nextDouble());
				}
				index.set(e.index());
				featureVector.set(row);
				writer.append(index, featureVector);
			}
		} finally {
			Closeables.close(writer, false);
		}
	}

	@SuppressWarnings("deprecation")
	private static long runSolver(Path ratings, Path output, Path pathToUorM, 
			int currentIteration, String matrixName, int numEntities, 
			Configuration conf, FileSystem fs) throws Exception {
		// necessary for local execution in the same JVM only
		SharingMapper.reset();

		Job job = createJob(conf, fs, ratings, output);
		job.setInputFormatClass(SequenceFileInputFormat.class);
		job.setMapperClass(MultithreadedSharingMapper.class);
		job.setMapOutputKeyClass(IntWritable.class);
		job.setMapOutputValueClass(VectorWritable.class);
		job.setOutputKeyClass(IntWritable.class);
		job.setOutputValueClass(VectorWritable.class);
		job.setOutputFormatClass(SequenceFileOutputFormat.class);
		job.setNumReduceTasks(0);

		Configuration solverConf = job.getConfiguration();
		solverConf.set("lambda", String.valueOf(lambda));
		solverConf.setInt("num_features", numFeatures);
		solverConf.set("num_entities", String.valueOf(numEntities));

		FileStatus[] parts = fs.listStatus(pathToUorM, PathFilters.partFilter());
		for (FileStatus part : parts) {
			DistributedCache.addCacheFile(part.getPath().toUri(), solverConf);
		}

		MultithreadedMapper.setMapperClass(job, SolveExplicitFeedbackMapper.class);
		MultithreadedMapper.setNumberOfThreads(job, numThreadsPerSolver);

		long res = runJob(job);
		return res;
	}
	
	public static long runEvaluator(Configuration conf, FileSystem fs) throws Exception {
		Path errors = new Path(root+"/error");
		Path testPath = new Path(root+"/ratings.probe");

		Job job = createJob(conf, fs, testPath, errors);
		job.setMapperClass(PredictRatingsMapper.class);
		job.setMapOutputKeyClass(DoubleWritable.class);
		job.setMapOutputValueClass(NullWritable.class);
		job.setOutputKeyClass(DoubleWritable.class);
		job.setOutputValueClass(NullWritable.class);
		job.setNumReduceTasks(0);
		job.setOutputFormatClass(SequenceFileOutputFormat.class);
		
		Configuration evaluatorConf = job.getConfiguration();
		evaluatorConf.set("userFeatures", root+"/U");
		evaluatorConf.set("itemFeatures", root+"/M");

		long res = runJob(job);

		BufferedWriter writer  = null;
		try {
			FSDataOutputStream outputStream = fs.create(new Path(root+"/rmse.txt"));
			double rmse = computeRmse(errors, conf);
			writer = new BufferedWriter(new OutputStreamWriter(outputStream, Charsets.UTF_8));
			writer.write(String.valueOf(rmse));
		} finally {
			Closeables.close(writer, false);
		}
		return res;
	}
	
	static double computeRmse(Path errors, Configuration conf) {
    	RunningAverage average = new FullRunningAverage();
		for (Pair<DoubleWritable,NullWritable> entry
			: new SequenceFileDirIterable<DoubleWritable, NullWritable>(errors, PathType.LIST, PathFilters.logsCRCFilter(),
			conf)) {
			DoubleWritable error = entry.getFirst();
			average.addDatum(error.get() * error.get());
		}

		return Math.sqrt(average.getAverage());
	}
	
	public static class PredictRatingsMapper extends Mapper<LongWritable,Text,DoubleWritable,NullWritable> {

		private OpenIntObjectHashMap<Vector> U;
		private OpenIntObjectHashMap<Vector> M;

		private final DoubleWritable error = new DoubleWritable();

		@Override
		protected void setup(Context ctx) throws IOException, InterruptedException {
			Configuration conf = ctx.getConfiguration();

			Path pathToU = new Path(conf.get("userFeatures"));
			Path pathToM = new Path(conf.get("itemFeatures"));

			U = readMatrixByRows(pathToU, conf);
			M = readMatrixByRows(pathToM, conf);
		}

		@Override
		protected void map(LongWritable key, Text value, Context ctx) throws IOException, InterruptedException {
			String[] tokens = value.toString().split(",");
			int userID = Integer.parseInt(tokens[0]);
			int itemID = Integer.parseInt(tokens[1]);
			float rating = Float.parseFloat(tokens[2]);

			if (U.containsKey(userID) && M.containsKey(itemID)) {
				double estimate = U.get(userID).dot(M.get(itemID));
				error.set(rating - estimate);
				ctx.write(error, NullWritable.get());
			}
		}
		
		protected OpenIntObjectHashMap<Vector> readMatrixByRows(Path dir, Configuration conf) {
			OpenIntObjectHashMap<Vector> matrix = new OpenIntObjectHashMap<Vector>();
			for (Pair<IntWritable,VectorWritable> pair
					: new SequenceFileDirIterable<IntWritable,VectorWritable>(dir, PathType.LIST, PathFilters.partFilter(), conf)) {
				int rowIndex = pair.getFirst().get();
				Vector row = pair.getSecond().get();
				matrix.put(rowIndex, row);
			}
			return matrix;
		}
	}
}
